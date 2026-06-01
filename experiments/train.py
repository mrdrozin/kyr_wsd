"""
Обучение WSD-модели по YAML-конфигу и seed'у.

Число эпох НЕ назначается вручную: обучение идёт до max_epochs с early stopping
по dev accuracy (held-out часть train, не пересекается с тестом). Это снимает
претензию «отладка на тестовой выборке» и «почему столько эпох».

Метрика отбора — dev accuracy, а не dev loss: для задачи ранжирования смыслов
NLL-loss может расти из-за переуверенности модели даже при хорошем ранжировании.

Запуск (обычно на Colab с GPU):
    python experiments/train.py --config experiments/configs/kyrgyzbert.yaml --seed 42
    python experiments/train.py --config experiments/configs/xlmr.yaml --seed 42 --lora

Результат — каталог experiments/runs/{model}_{full|lora}_seed{seed}/:
    model.pt           веса лучшей по dev accuracy модели
    run_config.json    конфиг прогона (для evaluate.py)
    loss_history.json  история train/dev лоссов и dev accuracy
    metrics.json       лучшая эпоха и dev-метрики
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# fast-токенизаторы используют внутренний пул потоков; при DataLoader-воркерах
# это даёт thread-thrashing — отключаем явно
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from model import (GlossSelectionModel, GroupedWSDDataset, apply_lora,
                   make_collate_fn, run_eval)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "experiments/data"
RUNS = ROOT / "experiments/runs"


def load_config(path):
    """base.yaml + оверрайд конкретной модели."""
    base = yaml.safe_load((Path(path).parent / "base.yaml").read_text())
    override = yaml.safe_load(Path(path).read_text())
    base.update(override or {})
    return base


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch, device):
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    if "token_type_ids" in batch:
        inputs["token_type_ids"] = batch["token_type_ids"].to(device)
    inputs["labels"] = [t.to(device) for t in batch["labels"]]
    inputs["counts"] = batch["counts"]
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--data-prefix", default="",
                        help="суффикс для train/dev файлов (например, 'extended' -> "
                             "data/train_extended.json). Default — без суффикса.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = pick_device()

    suffix = f"_{args.data_prefix}" if args.data_prefix else ""
    mode = "lora" if args.lora else "full"
    model_short = cfg["model_name"].split("/")[-1]
    run_dir = RUNS / f"{model_short}_{mode}{suffix}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    lr = cfg["lora"]["lr"] if args.lora else cfg["lr"]
    print(f"[{run_dir.name}] device={device} lr={lr} batch={cfg['batch_size']}")

    # --- данные
    train_rows = json.loads((DATA / f"train{suffix}.json").read_text(encoding="utf-8"))
    dev_rows = json.loads((DATA / f"dev{suffix}.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    collate = make_collate_fn(tokenizer, max_length=cfg["max_length"])

    # num_workers — токенизация в фоновых процессах, чтобы GPU не простаивал;
    # persistent_workers=True — не пересоздавать воркеров на каждой эпохе.
    # 4 — комфортно при batch 32–64 и ~5 глоссах на инстанс на Colab A100 (12 vCPU);
    # если на T4 / мало RAM — снизьте до 2.
    loader_kwargs = dict(batch_size=cfg["batch_size"], collate_fn=collate,
                         num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(GroupedWSDDataset(train_rows), shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(GroupedWSDDataset(dev_rows), shuffle=False, **loader_kwargs)

    # --- модель
    model = GlossSelectionModel(cfg["model_name"], dropout=cfg["dropout"])
    if args.lora:
        model = apply_lora(
            model,
            r=cfg["lora"]["r"], alpha=cfg["lora"]["alpha"],
            dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"].get("target_modules", ["query", "value"]),
        )
    model.to(device)

    # --- оптимизатор и расписание
    grad_accum = cfg.get("grad_accum", 1)
    steps_per_epoch = max(len(train_loader) // grad_accum, 1)
    total_steps = steps_per_epoch * cfg["max_epochs"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=cfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg["warmup_ratio"]), total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train": [], "dev": []}
    best_acc, best_epoch, patience = -1.0, -1, 0
    global_step = 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        running, seen = 0.0, 0
        for i, batch in enumerate(train_loader):
            inputs = move_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = model(**inputs)["loss"] / grad_accum
            scaler.scale(loss).backward()
            running += float(loss) * grad_accum
            seen += 1

            if (i + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                if global_step % cfg.get("log_steps", 50) == 0:
                    history["train"].append(
                        {"step": global_step, "loss": running / seen})
                    running, seen = 0.0, 0

        dev = run_eval(model, dev_loader, device)
        history["dev"].append({"step": global_step, "epoch": epoch,
                               "loss": dev["loss"], "accuracy": dev["accuracy"]})
        print(f"  epoch {epoch}: dev_loss={dev['loss']:.4f} "
              f"dev_acc={dev['accuracy']:.4f}")

        if dev["accuracy"] > best_acc:
            best_acc, best_epoch, patience = dev["accuracy"], epoch, 0
            torch.save(model.state_dict(), run_dir / "model.pt")
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                print(f"  early stopping (patience {patience})")
                break

    run_config = {
        "model_name": cfg["model_name"], "lora": args.lora, "seed": args.seed,
        "max_length": cfg["max_length"], "dropout": cfg["dropout"], "lr": lr,
    }
    if args.lora:
        run_config["lora_params"] = cfg["lora"]
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "loss_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(
        {"best_epoch": best_epoch, "best_dev_accuracy": best_acc},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{run_dir.name}] best dev_acc={best_acc:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()

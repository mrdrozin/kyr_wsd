"""
Архитектура WSD-модели для киргизского — model-agnostic вариант.

Подход — gloss selection (Huang et al. 2019, GlossBERT; Yap et al. 2020): инстанс
это предложение с выделенным словом и список глоссов-кандидатов; модель оценивает
каждую пару (контекст, глосс) и через softmax по группе выбирает верный смысл.
Обучение — NLL по группе кандидатов.

Энкодер берётся через AutoModel, поэтому одна и та же архитектура работает для
KyrgyzBERT, mBERT, XLM-RoBERTa и BERTurk — это даёт честное сравнение бейзлайнов
(меняется только энкодер). Отсутствие token_type_ids у XLM-RoBERTa обрабатывается
корректно: collate отдаёт их только если токенизатор их производит.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel


class GroupedWSDDataset(Dataset):
    """Группирует строки instance_id/context/gloss/label в инстансы-группы."""

    def __init__(self, rows):
        groups = {}
        for r in rows:
            g = groups.setdefault(r["instance_id"], {
                "instance_id": r["instance_id"],
                "contexts": [], "glosses": [], "labels": [],
            })
            g["contexts"].append(r["context"])
            g["glosses"].append(r["gloss"])
            g["labels"].append(int(r["label"]))
        self.groups = list(groups.values())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]


def make_collate_fn(tokenizer, max_length=256):
    """Возвращает collate-функцию. token_type_ids включаются, только если
    токенизатор их производит (у XLM-RoBERTa их нет)."""
    use_token_type = "token_type_ids" in tokenizer("a", "b")

    def collate(batch):
        input_ids, attention, token_type = [], [], []
        labels, counts, instance_ids = [], [], []
        for g in batch:
            for ctx, gloss in zip(g["contexts"], g["glosses"]):
                enc = tokenizer(
                    ctx, gloss,
                    add_special_tokens=True, max_length=max_length,
                    padding="max_length", truncation=True, return_tensors="pt",
                )
                input_ids.append(enc["input_ids"][0])
                attention.append(enc["attention_mask"][0])
                if use_token_type:
                    token_type.append(enc["token_type_ids"][0])
            labels.append(torch.tensor(g["labels"], dtype=torch.long))
            counts.append(len(g["contexts"]))
            instance_ids.append(g["instance_id"])
        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention),
            "labels": labels,
            "counts": counts,
            "instance_ids": instance_ids,
        }
        if use_token_type:
            out["token_type_ids"] = torch.stack(token_type)
        return out

    return collate


class GlossSelectionModel(nn.Module):
    """Энкодер + линейная голова, выдающая один скор на пару (контекст, глосс)."""

    def __init__(self, model_name, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                labels=None, counts=None, **kwargs):
        enc_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            enc_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**enc_kwargs)

        # вектор первого токена ([CLS] / <s>) как представление пары
        cls = outputs.last_hidden_state[:, 0, :]
        scores = self.score(self.dropout(cls)).squeeze(-1)

        loss = None
        if labels is not None and counts is not None:
            total, start = 0.0, 0
            for n, group_labels in zip(counts, labels):
                log_probs = torch.log_softmax(scores[start:start + n], dim=0)
                true_idx = int(torch.argmax(group_labels))
                total = total + (-log_probs[true_idx])
                start += n
            loss = total / len(counts)

        return {"loss": loss, "scores": scores}


def apply_lora(model, r=16, alpha=32, dropout=0.1, target_modules=("query", "value")):
    """Навешивает LoRA на энкодер; scoring-голова остаётся полностью обучаемой.

    target_modules — суффиксы имён линейных слоёв энкодера. ["query","value"] —
    минимально, только attention Q/V; ["query","key","value","dense"] —
    шире, включая выходные и FFN-линейные слои (имя "dense" в BERT/RoBERTa
    совпадает с тремя разными слоями: attention.output.dense, intermediate.dense,
    output.dense — это даёт заметно больше ёмкости адаптера)."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    model.encoder = get_peft_model(model.encoder, config)
    return model


@torch.no_grad()
def run_eval(model, loader, device):
    """Прогон по сгруппированной выборке. Возвращает loss, accuracy и
    correct (0/1 на инстанс) для bootstrap-оценки доверительных интервалов.
    Инстансы без верной метки (label==0 у всех) пропускаются."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    correct, instance_ids, skipped = [], [], 0

    for batch in loader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"].to(device)

        labels = [t.to(device) for t in batch["labels"]]
        out = model(**inputs, labels=labels, counts=batch["counts"])
        if out["loss"] is not None:
            total_loss += float(out["loss"])
            n_batches += 1

        scores = out["scores"].detach().cpu()
        start = 0
        for n, group_labels, iid in zip(batch["counts"], batch["labels"],
                                        batch["instance_ids"]):
            group_scores = scores[start:start + n]
            start += n
            if int(group_labels.sum()) == 0:   # нет верного варианта
                skipped += 1
                continue
            pred = int(torch.argmax(group_scores))
            true = int(torch.argmax(group_labels))
            correct.append(int(pred == true))
            instance_ids.append(iid)

    n = len(correct)
    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": sum(correct) / max(n, 1),
        "correct": correct,
        "instance_ids": instance_ids,
        "skipped": skipped,
    }

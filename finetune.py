#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Шаг 2. QLoRA-файнтюн Qwen2.5-1.5B-Instruct на диалогах.
Запуск: python finetune.py
Выход: out-adapter/ (LoRA-веса — они и есть ТВОЯ нейросеть)
Время: ~30-40 мин на RTX 3050 (40k пар, 1 эпоха).
Если время поджимает — уменьши MAX_STEPS ниже (например 800 вместо 1500)."""
import json
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model

# ------- настройки -------
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_STEPS = 1500       # ~30-40 мин на 3050. Мало времени? поставь 800
MAX_LEN = 512
LR = 2e-4
# -----------------------

assert torch.cuda.is_available(), "НЕТ CUDA! Проверь драйвер: nvidia-smi"
print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} ГБ", flush=True)

tok = AutoTokenizer.from_pretrained(BASE)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
model.config.use_cache = False

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["messages"])
    return out

train_msgs = load_jsonl("data/train.jsonl")
val_msgs = load_jsonl("data/val.jsonl")
print(f"обучающих: {len(train_msgs):,}, валидация: {len(val_msgs)}", flush=True)

class ChatDS(Dataset):
    def __init__(self, msgs):
        self.items = []
        for m in msgs:
            full = tok.apply_chat_template(m, tokenize=False)
            prefix = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
            full_ids = tok(full, truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]
            pref_ids = tok(prefix, truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]
            n = min(len(pref_ids), len(full_ids))
            self.items.append((full_ids, n))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        ids, n = self.items[i]
        labels = [-100] * n + ids[n:]
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

def collate(batch):
    mx = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id or tok.eos_token_id
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        d = mx - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad] * d)
        out["labels"].append(b["labels"] + [-100] * d)
        out["attention_mask"].append(b["attention_mask"] + [0] * d)
    return {k: torch.tensor(v) for k, v in out.items()}

args = TrainingArguments(
    output_dir="out-adapter", per_device_train_batch_size=2, gradient_accumulation_steps=8,
    learning_rate=LR, max_steps=MAX_STEPS, warmup_steps=50, logging_steps=20,
    eval_strategy="steps", eval_steps=250, eval_on_start=False,
    fp16=True, optim="paged_adamw_8bit", save_strategy="no", report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=ChatDS(train_msgs),
                  eval_dataset=ChatDS(val_msgs), data_collator=collate)
trainer.train()

model.save_pretrained("out-adapter")
tok.save_pretrained("out-adapter")
print("\nLoRA-веса сохранены в out-adapter/ — это ТВОЯ часть модели", flush=True)

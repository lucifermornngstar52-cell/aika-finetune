#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Шаг 3. Мердж LoRA с базой + быстрый тест болтовни.
Запуск: python merge_test.py
Выход: merged/ (полная модель ~3 ГБ) + примеры ответов в консоль.
Потом её конвертим в GGUF для телефона (см. README)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"

print("загружаю базу в fp16...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map={"": 0})

print("прикручиваю твою LoRA...", flush=True)
model = PeftModel.from_pretrained(model, "out-adapter")
model = model.merge_and_unload()
model.save_pretrained("merged", safe_serialization=True, max_shard_size="2GB")
tok.save_pretrained("merged")
print("модель сохранена в merged/", flush=True)

# --- тест ---
model.eval()
def ask(msg):
    ids = tok.apply_chat_template([{"role": "user", "content": msg}], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=60, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

print("\n=== ТЕСТ ТВОЕЙ НЕЙРОСЕТИ ===", flush=True)
for q in ["привет", "как дела?", "ты кто такая?", "что делаешь?"]:
    print(f"Ю: {q}\nА: {ask(q)}\n", flush=True)

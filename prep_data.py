#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Шаг 1. Скачивает диалоги и готовит train.jsonl/val.jsonl.
Запуск: python prep_data.py
Выход: data/train.jsonl, data/val.jsonl (формат ChatML-сообщений для Qwen)"""
import json
import os
import random
import urllib.request

random.seed(42)
os.makedirs("data", exist_ok=True)

# датасет уже готов (приехал в архиве) — ничего не делаем
if os.path.exists("data/train.jsonl") and os.path.exists("data/val.jsonl"):
    print("датасет уже готов: data/train.jsonl, data/val.jsonl — шаг пропущен")
    raise SystemExit(0)

URL = "https://huggingface.co/api/datasets/Den4ikAI/russian_dialogues/parquet/default/train/0.parquet"
PARQUET = "data/raw.parquet"

# --- скачивание (275 МБ, ~2-10 мин в зависимости от интернета) ---
if not os.path.exists(PARQUET):
    print("качаю датасет (~275 МБ)...", flush=True)
    urllib.request.urlretrieve(URL, PARQUET)
print("датасет скачан", flush=True)

import pyarrow.parquet as pq

t = pq.read_table(PARQUET, columns=["question", "answer", "relevance"])
qs, ans, rel = (t.column(c).to_pylist() for c in ("question", "answer", "relevance"))

pairs = []
seen = set()
for q, a, r in zip(qs, ans, rel):
    if r != 1:
        continue
    q = " ".join(str(q).split())
    a = " ".join(str(a).split())
    if not (2 < len(q) < 200 and 1 < len(a) < 200):
        continue
    key = (q.lower(), a.lower())
    if key in seen:
        continue
    seen.add(key)
    pairs.append((q, a))

random.shuffle(pairs)
N = min(40000, len(pairs))  # 40k пар = ~1 эпоха в бюджет 90 минут
train, val = pairs[:N], pairs[N:N + 500]
print(f"всего чистых пар: {len(pairs):,}, беру {N:,} на обучение", flush=True)

def dump(path, chunk):
    with open(path, "w", encoding="utf-8") as f:
        for q, a in chunk:
            f.write(json.dumps({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]}, ensure_ascii=False) + "\n")

dump("data/train.jsonl", train)
dump("data/val.jsonl", val)
print(f"готово: data/train.jsonl ({N:,}), data/val.jsonl (500)")

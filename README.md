# ⚡ Аика-Файнтюн: своя нейросеть за 90 минут

Qwen2.5-1.5B-Instruct + QLoRA на твоих диалогах = **твоя модель**.
База — заготовка (дерево), LoRA — твоя резьба. Без LoRA это Qwen, с LoRA — это Аика.

## Тайминг вторника (90 минут)

| Мин | Шаг |
|---|---|
| 0-10 | Подготовка окружения (шаг 0) |
| 10-20 | `python prep_data.py` (качает датасет) |
| 20-30 | Качается база (3.1 ГБ) + старт файнтюна |
| 30-65 | `python finetune.py` — идёт обучение (контроль по логам) |
| 65-75 | `python merge_test.py` — мердж + тест болтовни |
| 75-90 | GGUF-конвертация для телефона (шаг 4) |

## Шаг 0. Окружение (один раз)

Проверка GPU (должна показать 3050):
```
nvidia-smi
```

Python 3.11 если нет:
```
winget install Python.Python.3.11
```

В папке проекта:
```
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Шаги 1-3

```
python prep_data.py     # датасет -> data/train.jsonl
python finetune.py      # LoRA -> out-adapter/  (~30-40 мин)
python merge_test.py    # merged/ + тест ответов в консоли
```

**Если обучение не успевает:** открой `finetune.py`, поставь `MAX_STEPS = 800` (или меньше). Качество чуть хуже, но модель будет.
**Если интернет слабый:** сначала запусти `prep_data.py` и не трогай, пока качает — всё остальное можно параллельно.

## Шаг 4. GGUF для телефона (модель ~1 ГБ, работает офлайн)

```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
pip install -r requirements.txt
python convert_hf_to_gguf.py ../merged --outfile ../aika-1.5b-f16.gguf --outtype f16
python gguf-py/gguf/scripts/gguf_convert.py ../aika-1.5b-f16.gguf q4_k_m
```
(вторая команда делает из 3 ГБ готовый ~1 ГБ файл `aika-1.5b-q4_k_m.gguf`)

На телефон:
1. Ставишь **PocketPal AI** (Play Market, бесплатно)
2. Кидаешь `aika-1.5b-q4_k_m.gguf` в телефон
3. В PocketPal: + Add model → выбираешь файл → Load
4. Пишешь «привет» — отвечает офлайн. Это она, твоя.

## Troubleshooting

- **CUDA out of memory**: в `finetune.py` уменьши `per_device_train_batch_size` до 1, а `gradient_accumulation_steps` увеличь до 16
- **bitsandbytes не ставится**: `pip install bitsandbytes --prefer-binary`
- **Нет CUDA**: nvidia-smi пустой → драйвер NVIDIA не установлен, дальше смысла нет
- **prep_data висит на скачке**: школьный интернет — качай датасет частями или с телефона-хотспота

## Что потом

Файл `out-adapter/` — твои LoRA-веса (30-50 МБ). Их можно:
- докачивать на других данных (учебники, мемы, свой стиль речи)
- выложить в GitHub как «весы своей нейросети»
- применить к другой базе (Qwen2.5-3B, 7B на более мощной GPU)

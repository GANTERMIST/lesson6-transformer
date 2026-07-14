# Домашнее задание: Генератор текста на базе Transformer (decoder-only)

Решение задания из [AKlimovUrfu/lesson6-transformer](https://github.com/AKlimovUrfu/lesson6-transformer)
(`HOMEWORK.md`).

## Что реализовано

- `GeneratorTransformer` — decoder-only Transformer "с нуля" (causal self-attention,
  позиционные эмбеддинги, 4 decoder-блока, tied embeddings) — `model.py`
- Датасет со скользящим окном по **всему** тексту, деление на абзацы, `<bos>/<eos>` — `dataset.py`
- Обучение с mixed precision (`torch.amp.autocast` + `GradScaler`), `batch_size=1` — `train.py`
- Авторегрессивная генерация со сдвигом контекста на 1 токен влево — `generate.py`
- Beam search (дополнительное задание) — метод `GeneratorTransformer.beam_search_generate`
- Чат-интерфейс (`chat()` с циклом `input()/quit`, как в `HOMEWORK.md`) — `chat.py`
- Полный воспроизводимый прогон — `solution.ipynb` (ячейки запускаются по порядку)

## Данные и токенизатор

- Корпус: 144696 символов, книга, скачанная с Project Gutenberg
- Токенизатор: BPE, обучен на этом корпусе (`tokenizers` BPE, vocab_size=3449)
- Контекст: 128 токенов
- Обучающих примеров (скользящих окон): 316

## Обучение

- Эпох: 3, batch_size=1, learning_rate=1e-4
- Число параметров модели: 4.04M
- Средний loss: эпоха 1 = 28.0646 -> последняя эпоха = 9.4585
- Полный лог обучения: [`training_log.txt`](./training_log.txt)
- График loss по эпохам: [`training_loss.png`](./training_loss.png)

## Примеры генерации (реальный вывод обученной модели)

См. [`examples/generation_samples.txt`](./examples/generation_samples.txt) — там
приведены примеры как обычной сэмплированной генерации, так и beam search.

## Как запустить

```bash
pip install torch tokenizers
python train.py --corpus corpus.txt --epochs 3
python generate.py --prompt "Once upon a time" --temperature 0.8
python generate.py --prompt "Once upon a time" --beam --beam-width 4
python chat.py
```

Либо открыть `solution.ipynb` в Google Colab и выполнить ячейки по порядку.

## Структура репозитория

```
homework/
  README.md                     — этот файл
  solution.ipynb                — полный воспроизводимый ноутбук
  model.py                      — архитектура GeneratorTransformer
  dataset.py                    — подготовка данных (скользящее окно)
  train.py                      — обучение
  generate.py                   — генерация (сэмплирование / beam search)
  chat.py                       — интерактивный чат
  checkpoint.pt                 — обученные веса
  tokenizer.json                — обученный BPE-токенизатор
  corpus.txt                    — текст, на котором обучалась модель
  training_log.txt              — лог обучения (loss по шагам/эпохам)
  training_loss.png             — график обучения
  examples/generation_samples.txt — примеры генерации
```

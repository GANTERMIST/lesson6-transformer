'''
dataset.py — подготовка текстового корпуса: разбиение на абзацы,
оборачивание в <bos>/<eos> и нарезка скользящим окном по всему тексту.
'''
import torch
from torch.utils.data import Dataset


def split_into_blocks(text):
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if len(p) > 0]


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, pad_id, bos_id, eos_id, max_length=128, stride=None):
        stride = stride or max_length
        self.examples = []

        all_ids = []
        for block in split_into_blocks(text):
            ids = tokenizer.encode(block).ids
            all_ids.append(bos_id)
            all_ids.extend(ids)
            all_ids.append(eos_id)

        for start in range(0, max(1, len(all_ids) - 1), stride):
            chunk = all_ids[start:start + max_length + 1]
            if len(chunk) < 2:
                continue
            if len(chunk) < max_length + 1:
                chunk = chunk + [pad_id] * (max_length + 1 - len(chunk))
            self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

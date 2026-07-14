'''
train.py — обучение GeneratorTransformer на текстовом корпусе.

Запуск:
    python train.py --corpus corpus.txt --epochs 3 --context-len 128
'''
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from model import GeneratorTransformer
from dataset import TextDataset

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def build_tokenizer(text, vocab_size=8000, save_path="tokenizer.json"):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, min_frequency=2)
    tok.train_from_iterator([text], trainer=trainer)
    tok.save(save_path)
    return tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="corpus.txt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = build_tokenizer(text, vocab_size=args.vocab_size)
    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    dataset = TextDataset(text, tokenizer, pad_id, bos_id, eos_id, max_length=args.context_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    config = dict(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=256, n_heads=8, n_layers=4, d_ff=1024,
        context_len=args.context_len,
        pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id,
    )
    model = GeneratorTransformer(tokenizer=tokenizer, device=device, **config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (xb, yb) in enumerate(dataloader, start=1):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(dataloader)} loss={loss.item():.4f}")

        print(f"=== Epoch {epoch} avg loss = {total_loss / max(1, len(dataloader)):.4f} ===")
        model.save_checkpoint(args.checkpoint, config)

    print("Готово. Чекпоинт:", args.checkpoint)


if __name__ == "__main__":
    main()

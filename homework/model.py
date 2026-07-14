'''
model.py — архитектура GeneratorTransformer (decoder-only Transformer).

Домашнее задание, урок 6: генератор текста, обучающийся авторегрессивно
предсказывать следующий токен. Модуль самодостаточен и не зависит от
остального ноутбука — его можно импортировать из train.py / generate.py / chat.py.
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GeneratorTransformer(nn.Module):
    '''Decoder-only Transformer для авторегрессивной генерации текста.'''

    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        context_len=128,
        dropout=0.1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tokenizer=None,
        device="cpu",
    ):
        super().__init__()
        self.context_len = context_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer
        self.device = device

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(2048, context_len))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # tied weights

    def forward(self, input_ids):
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt, context_len=None, temperature=1.0, max_out_tokens=200):
        '''Авторегрессивная генерация со сдвигом контекста на 1 токен влево.'''
        self.eval()
        context_len = context_len or self.context_len

        input_ids = self.tokenizer.encode(prompt).ids
        input_ids = [self.bos_token_id] + input_ids
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated = input_ids.clone()
        for _ in range(max_out_tokens):
            model_input = generated[:, -context_len:]
            logits = self(model_input)
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == self.eos_token_id:
                break

        return self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

    @torch.no_grad()
    def beam_search_generate(self, prompt, beam_width=4, max_out_tokens=80, context_len=None):
        '''Beam search генерация (дополнительное задание).'''
        self.eval()
        context_len = context_len or self.context_len
        tok = self.tokenizer

        start_ids = [self.bos_token_id] + tok.encode(prompt).ids
        beams = [(start_ids, 0.0, False)]

        for _ in range(max_out_tokens):
            candidates = []
            for seq, score, finished in beams:
                if finished:
                    candidates.append((seq, score, finished))
                    continue
                model_input = torch.tensor([seq[-context_len:]], dtype=torch.long, device=self.device)
                logits = self(model_input)
                log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
                for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                    new_seq = seq + [tid]
                    new_score = score + lp
                    candidates.append((new_seq, new_score, tid == self.eos_token_id))

            candidates.sort(key=lambda c: c[1] / len(c[0]), reverse=True)
            beams = candidates[:beam_width]
            if all(f for _, _, f in beams):
                break

        return tok.decode(beams[0][0], skip_special_tokens=True)

    def save_checkpoint(self, path, config):
        torch.save({"model_state_dict": self.state_dict(), "config": config}, path)

    @classmethod
    def load_from_checkpoint(cls, path, tokenizer, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        config = ckpt["config"]
        model = cls(tokenizer=tokenizer, device=device, **config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return model

'''
generate.py — генерация текста обученной моделью (простая или beam search).

Запуск:
    python generate.py --prompt "Once upon a time" --temperature 0.8
    python generate.py --prompt "Once upon a time" --beam --beam-width 4
'''
import argparse

import torch
from tokenizers import Tokenizer

from model import GeneratorTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--beam", action="store_true", help="использовать beam search вместо сэмплирования")
    parser.add_argument("--beam-width", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = GeneratorTransformer.load_from_checkpoint(args.checkpoint, tokenizer=tokenizer, device=device)

    if args.beam:
        output = model.beam_search_generate(args.prompt, beam_width=args.beam_width, max_out_tokens=args.max_tokens)
    else:
        output = model.generate(args.prompt, temperature=args.temperature, max_out_tokens=args.max_tokens)

    print(output)


if __name__ == "__main__":
    main()

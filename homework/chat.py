'''
chat.py — интерактивный чат с обученной моделью (раздел 7 HOMEWORK.md).

Запуск:
    python chat.py
'''
import torch
from tokenizers import Tokenizer

from model import GeneratorTransformer


def chat():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("tokenizer.json")
    model = GeneratorTransformer.load_from_checkpoint("checkpoint.pt", tokenizer=tokenizer, device=device)
    model.eval()

    while True:
        user_input = input("Вы: ")
        if user_input.lower() == "quit":
            break

        response = model.generate(user_input, temperature=0.8, max_out_tokens=100)
        print(f"Бот: {response}")


if __name__ == "__main__":
    chat()

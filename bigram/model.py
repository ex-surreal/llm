import os
from os import makedirs
import sys
import random
import torch
import torch.nn as nn
from bigram.data import DataHandler

from bigram.encoder import TokenEncoder
from bigram.tokenizer import tokenize

class BigramLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table: nn.Module = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)

        if targets == None:
            return logits, None

        b, t, c = logits.shape

        logits = logits.view(b*t, c)
        targets = targets.view(b*t)
        loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self.forward(index)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

def train(train_src: str, dest: str, max_iter: int, checkpoint: int):
    makedirs(dest, exist_ok=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    handler = DataHandler(train_src, device, f"{dest}/encoder.json")
    model = BigramLM(handler.encoder.size)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(max_iter):
        xb, yb = handler.get_batch("train")
        _, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (i+1) % checkpoint == 0:
            print(f"Iteration {i} complete {loss.item()}")
            if os.path.isfile(f"{dest}/checkpoint"):
                print("Checkpointing ...")
                torch.save(model.state_dict(), f"{dest}/model")
                os.remove(f"{dest}/checkpoint")
    torch.save(model.state_dict(), f"{dest}/model")

def talk(src: str):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    encoder = TokenEncoder.load(f"{src}/encoder.json")
    model = BigramLM(encoder.size)
    model.load_state_dict(torch.load(f"{src}/model"))
    model.to(device)
    while True:
        print(">")
        line = sys.stdin.readline()
        tokens = tokenize(line)
        input = torch.tensor([encoder.encode(tokens)], dtype=torch.long, device=device)
        output = model.generate(input, max_tokens=random.randint(0, 100))[0].tolist()
        print("".join(encoder.decode(output)[len(tokens):]))

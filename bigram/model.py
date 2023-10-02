import os
from os import makedirs
import sys
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from bigram.data import DataHandler

from bigram.encoder import TokenEncoder
from bigram.param import HyperParam
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

class FeedForward(nn.Module):
    def __init__(self, param: HyperParam) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param.n_embd, 4*param.n_embd),
            nn.ReLU(),
            nn.Linear(4*param.n_embd, param.n_embd),
            nn.Dropout(param.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    def __init__(self, param: HyperParam) -> None:
        super().__init__()
        head_size = param.n_embd // param.n_head
        self.key = nn.Linear(param.n_embd, head_size, bias=False)
        self.query = nn.Linear(param.n_embd, head_size, bias=False)
        self.value = nn.Linear(param.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(param.block_size, param.block_size)))
        self.dropout = nn.Dropout(param.dropout)

    def forward(self, x):
        _, t, _ = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, param: HyperParam) -> None:
        super().__init__()
        head_size = param.n_embd // param.n_head
        self.heads = nn.ModuleList([Head(param) for _ in range(param.n_head)])
        self.proj = nn.Linear(head_size * param.n_head, param.n_embd)
        self.dropout = nn.Dropout(param.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    def __init__(self, param: HyperParam) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(param)
        self.ffwd = FeedForward(param)
        self.ln1 = nn.LayerNorm(param.n_embd)
        self.ln2 = nn.LayerNorm(param.n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x+y)
        y = self.ffwd(x)
        x = self.ln2(x+y)
        return x

class GptLM(nn.Module):
    param: HyperParam

    def __init__(self, n_vocab: int, param: HyperParam) -> None:
        super().__init__()
        self.param = param
        self.token_embedding_table: nn.Module = nn.Embedding(n_vocab, param.n_embd)
        self.position_embedding_table: nn.Module = nn.Embedding(param.block_size, param.n_embd)
        self.blocks: nn.Module = nn.Sequential(*[Block(param) for _ in range(param.n_layer)])
        self.ln_f = nn.LayerNorm(param.n_embd)
        self.lm_head = nn.Linear(param.n_embd, n_vocab)
        self.device = param.device

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) or module.bias is not None:
                torch.nn.init.zeros_(module.bias)


    def forward(self, index, targets=None):
        _, t = index.shape
        tok_embds = self.token_embedding_table(index)
        pos_embds = self.position_embedding_table(torch.arange(t, device=self.device))
        x = tok_embds + pos_embds
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b*t, c)
            targets = targets.view(b*t)
            loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self.forward(index[:, -self.param.block_size:])
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index


def train(param_src: str, train_src: str, dest: str):
    makedirs(dest, exist_ok=True)
    param = HyperParam.create(param_src)
    print(f"HyperParameter: {param}")
    with DataHandler(train_src, dest, param) as handler:
        model = GptLM(handler.encoder.size, param)
        model.to(param.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=param.lr)
        print("Starting...")
        for i in tqdm(range(param.train_iter)):
            xb, yb = handler.get_batch("train")
            _, loss = model.forward(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if i % param.checkpoint == 0:
                print(f"Iteration {i} complete {loss.item()} {estimate_loss(model, handler, param.eval_iter)}")
                if os.path.isfile(f"{dest}/checkpoint"):
                    print("Checkpointing ...")
                    torch.save(model.state_dict(), f"{dest}/model")
                    os.remove(f"{dest}/checkpoint")
        torch.save(model.state_dict(), f"{dest}/model")

@torch.no_grad()
def estimate_loss(model: nn.Module, handler: DataHandler, eval_iter: int):
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iter)
        for i in range(eval_iter):
            x, y = handler.get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def talk(param_src: str, src: str):
    param = HyperParam.create(param_src)
    encoder = TokenEncoder.load(f"{src}/encoder.json")
    model = GptLM(encoder.size, param)
    model.load_state_dict(torch.load(f"{src}/model"))
    model.to(param.device)
    while True:
        print(">")
        line = sys.stdin.readline()[:-1]
        tokens = tokenize(line)
        input = torch.tensor(encoder.encode(tokens), dtype=torch.long, device=param.device)
        output = model.generate(input.unsqueeze(0), max_tokens=random.randint(0, 100))[0].tolist()
        print("".join(encoder.decode(output)[len(tokens):]))

import torch
from bigram.encoder import WordEncoder
from bigram.tokenizer import tokenize


class DataHandler:
    block_size: int
    batch_size: int
    fr: float

    def __init__(self, file_name: str, device, dest: str, block_size: int = 64, batch_size: int = 4, fr: float = 0.8) -> None:
        with open(file_name, "r", encoding="utf-8") as f:
            tokens = tokenize(f.read())
            self.encoder = WordEncoder.of_tokens(tokens)
            self.encoder.save(dest)
            self.data = torch.tensor(self.encoder.encode(tokens), dtype=torch.long)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.fr = fr

    def get_batch(self, name: str):
        n = int(len(self.data)*self.fr)
        work = self.data[:n] if name == "train" else self.data[n:]

        ix = torch.randint(len(work)-self.block_size, (self.batch_size,))

        return (
            torch.stack([work[i:i+self.block_size] for i in ix]).to(self.device),
            torch.stack([work[i+1:i+self.block_size+1] for i in ix]).to(self.device)
        )



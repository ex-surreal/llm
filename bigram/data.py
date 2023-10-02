import mmap
from os import path
import random
import torch
from bigram.encoder import TokenEncoder
from bigram.param import HyperParam
from bigram.tokenizer import tokenize


class DataHandler:
    root: str
    param: HyperParam

    def __init__(self, root: str, dest: str, param: HyperParam) -> None:
        print("Initialising DataHandler")
        self.root = root
        self.param = param
        self.encoder = TokenEncoder.of_files(root)
        self.encoder.save(path.join(dest, "encoder.json"))

    def __enter__(self):
        self.train = open(path.join(self.root, "train.txt"), 'rb')
        self.train_mm = mmap.mmap(self.train.fileno(), 0, access=mmap.ACCESS_READ)
        self.eval = open(path.join(self.root, "eval.txt"), 'rb')
        self.eval_mm = mmap.mmap(self.eval.fileno(), 0, access=mmap.ACCESS_READ)
        print("enter DataHandler")
        return self

    def __exit__(self, *_):
        self.train_mm.close()
        self.train.close()
        self.eval_mm.close()
        self.eval.close()
        print("exit DataHandler")
        return self

    def get_batch(self, name: str):
        mm = self.train_mm if name == "train" else self.eval_mm
        work_size = self.param.block_size * self.param.batch_size + 1
        size = 4 * (1 + work_size)
        start = random.randint(0, mm.size() - size)

        mm.seek(start)
        bs = mm.read(size)
        work_str = self._skip_non_utf_start(bs)[:work_size]
        work = torch.tensor(self.encoder.encode(tokenize(work_str)), dtype=torch.long)

        ix = torch.randint(len(work) - self.param.block_size, (self.param.batch_size,))

        return (
            torch.stack([work[i:i+self.param.block_size] for i in ix]).to(self.param.device),
            torch.stack([work[i+1:i+self.param.block_size+1] for i in ix]).to(self.param.device)
        )

    def _skip_non_utf_start(self, bs: bytes) -> str:
        c = 0
        for b in bs:
            if (b >> 6) != 0b10:
                break
            c += 1
        return bs[c:].decode(errors='ignore')

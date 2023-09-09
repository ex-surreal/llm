import torch
import torch.nn as nn

import sys
import random
import re

from torch.optim import Optimizer, optimizer

class CharEncoder:
    size: int
    char_to_id: dict
    id_to_char: dict

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.size = len(chars)
        self.char_to_id = {c:i for i, c in enumerate(chars)}
        self.id_to_char = {i:c for i, c in enumerate(chars)}

    def encode(self, s: str) -> list:
        return [self.char_to_id[x] for x in s]

    def decode(self, l: list) -> str:
        return "".join(self.id_to_char[x] for x in l)



if __name__ == "__main__":
    handler = DataHandler(sys.argv[1], device)

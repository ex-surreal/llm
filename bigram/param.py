from dataclasses import dataclass
import json

import torch

@dataclass
class HyperParam:
    train_split: float
    lr: float
    train_iter: int
    eval_iter: int
    checkpoint: int
    batch_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    device: torch.device

    @classmethod
    def create(cls, file_name: str):
        with open(file_name, 'r') as f:
            dict = json.load(f)
            instance = cls(**dict)
            instance.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            return instance

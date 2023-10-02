from dataclasses import dataclass
from glob import glob
import json
from os import path

from tqdm import tqdm

from bigram.tokenizer import tokenize


@dataclass
class TokenEncoder:
    size: int
    word_to_id: dict
    id_to_word: list

    def encode(self, tokens) -> list:
        return [self.word_to_id.get(x, self.size) for x in tokens]

    def decode(self, ids: list) -> list:
        return [self.id_to_word[x] if x < len(self.id_to_word) else "$$$$" for x in ids]

    def save(self, dest: str):
        with open(dest, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def of_files(cls, root):
        vocab = set()
        for file in glob(path.join(root, "*.txt")):
            f_size = path.getsize(file) / 1000000.0
            with open(file, 'r', encoding='utf-8') as f, tqdm(total=f_size) as t:
                for line in f:
                    vocab.update(set(tokenize(line)))
                    t.update(len(line.encode()) / 1000000.0)
        id_to_word = list(sorted(vocab))
        size = len(id_to_word)
        word_to_id = {c:i for i, c in enumerate(id_to_word)}
        return cls(size, word_to_id, id_to_word)

    @classmethod
    def load(cls, src: str):
        with open(src, "r") as f:
            dict = json.load(f)
            return cls(**dict)



import json


class WordEncoder:
    size: int
    word_to_id: dict
    id_to_word: list

    def __init__(self, size: int, word_to_id: dict, id_to_word: list) -> None:
        self.size = size
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word


    def encode(self, tokens: list) -> list:
        return [self.word_to_id.get(x, self.size) for x in tokens]

    def decode(self, l: list) -> list:
        print("decode", l)
        return [self.id_to_word[x] if x < len(self.id_to_word) else "$$$$" for x in l]

    def save(self, dest: str):
        with open(dest, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def of_tokens(cls, tokens: list):
        id_to_word = list(sorted(set(tokens)))
        size = len(id_to_word)
        word_to_id = {c:i for i, c in enumerate(id_to_word)}
        return cls(size, word_to_id, id_to_word)

    @classmethod
    def load(cls, src: str):
        with open(src, "r") as f:
            dict = json.load(f)
            return cls(**dict)



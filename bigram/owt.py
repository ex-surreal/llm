import lzma
from glob import glob
from os import path
from tqdm import tqdm

def extract(src: str, dst: str, train_fr: float):
    files = glob(path.join(src, "*.xz"))
    print(f"Total files: {len(files)}")
    n_train = int(len(files) * train_fr)
    _write_split(files[:n_train], path.join(dst, "train.txt"))
    _write_split(files[n_train:], path.join(dst, "eval.txt"))

def _write_split(files: list, dst: str):
    print(f"Writing {len(files)} to {dst}")
    with open(dst, 'w', encoding='utf-8') as out:
        for file in tqdm(files):
            with lzma.open(file, 'rt', encoding='utf-8') as f:
                out.write(f.read())

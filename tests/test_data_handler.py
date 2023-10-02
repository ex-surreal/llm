
from os import path
from shutil import rmtree
from tempfile import mkdtemp
import torch
from bigram.data import DataHandler
from bigram.param import HyperParam
from unittest import TestCase


param = HyperParam(
    block_size=5,
    batch_size=3,
    train_split=0.9,
    lr=3e-4,
    train_iter=10000,
    eval_iter=100,
    checkpoint=100,
    n_embd=384,
    n_head=8,
    n_layer=8,
    dropout=0.2,
    device=torch.device('cpu')
)

class TestDataHandler(TestCase):
    def setUp(self):
        print("HEY")
        self.root = mkdtemp()
        self.dest = mkdtemp()
        super().setUp()

    def tearDown(self):
        rmtree(self.root)
        rmtree(self.dest)
        super().tearDown()


    def test_batches(self):
        s = """This is a text.
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
£120 with special character © fine ˚ © ∂ ß å π Ò
"""
        self._setup_file(path.join(self.root, "train.txt"), s)
        self._setup_file(path.join(self.root, "eval.txt"), s)
        with DataHandler(self.root, self.dest, param) as test_obj:
            for _ in range(1000):
                x, y = test_obj.get_batch("train")
                x, y = x.tolist(), y.tolist()
                x = ["".join(test_obj.encoder.decode(p)) for p in x]
                y = ["".join(test_obj.encoder.decode(p)) for p in y]
                assert x[1:] == y[:-1]

    def _setup_file(self, name: str, content: str):
        with open(name, 'w', encoding='utf-8') as f:
            f.write(content)

import sys

from bigram.model import train

train(sys.argv[1], sys.argv[2], int(sys.argv[3]))

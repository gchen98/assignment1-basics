#from tests.test_train_bpe import test_train_bpe
from cs336_basics import bpe
from cs336_basics import io_utils

import time

special_tokens=['<|endoftext|>']

#workers=1
#input_path = "tests/fixtures/corpus.en"
#vocab_size = 500
#outmerges = "corpus_merges.txt"
#outvocab = "corpus_vocab.json"

#workers=4
#input_path = "data/TinyStoriesV2-GPT4-train.txt"
#input_path = "data/TinyStoriesV2-GPT4-valid.txt"
#vocab_size = 10000
#outmerges = "tinystories_merges.txt"
#outvocab = "tinystories_vocab.json"

workers=24
input_path = "data/owt_train.txt"
vocab_size = 32000
outmerges = "owt_merges.txt"
outvocab = "owt_vocab.json"


(vocab,merges) = bpe.run_train_bpe(input_path,vocab_size,special_tokens,workers)
io_utils.serialize_merges(outmerges,merges)
io_utils.serialize_vocab(outvocab,vocab)



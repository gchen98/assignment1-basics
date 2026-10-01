from functools import lru_cache
import json
import logging
import os
from typing import BinaryIO

# Create a logger specific to this file module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()

# Create a formatting layout
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to your logger
logger.addHandler(console_handler)


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

def serialize_merges(outfile,merge_list):
    bytes2unicode = gpt2_bytes_to_unicode()
    with open(outfile,"w",encoding='utf-8') as fout:
        for (merge0,merge1) in merge_list:
            remapped0 = [bytes2unicode[t] for t in list(merge0)]
            for b in remapped0:
                assert(not b.isspace())
                fout.write(b)
            fout.write(' ')
            remapped1 = [bytes2unicode[t] for t in list(merge1)]
            for b in remapped1:
                assert(not b.isspace())
                fout.write(b)
            fout.write('\n')


def deserialize_merges(infile):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(infile,"r",encoding = 'utf-8') as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]

    reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    # print(f"merges: {reference_merges}")
    return reference_merges


def serialize_vocab(vocab_file,vocab):

    bytes2unicode = gpt2_bytes_to_unicode()
    remapped_vocab = {}
    unique_terms = set()
    longest = 0
    longest_term = None
    for k,v in vocab.items():
        v2 = [bytes2unicode[t] for t in v]
        v2_len = len(v2)
        v2str = ''.join(v2)
        if v2_len> longest:
            longest = v2_len
            longest_term = v2str
        if v2str in unique_terms:
            logger.error(f"WARNING: {v2str} is already in the vocabulary!")
        unique_terms.add(v2str)
        #print(f"{v2str}")
        remapped_vocab[v2str] = k
    logger.info(f"Longest term is {longest_term}")
    assert(len(unique_terms)==len(vocab))
    with open(vocab_file,"w",encoding="utf-8") as f:
        json.dump(remapped_vocab,f,indent=4,ensure_ascii=False)
    #print(remapped_vocab)

def deserialize_vocab(vocab_file):
    reference_vocab_path = vocab_file
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    return reference_vocab

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """


    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    # print(f"File size {file_size} Chunk boundaries {chunk_boundaries}")
    return sorted(set(chunk_boundaries))
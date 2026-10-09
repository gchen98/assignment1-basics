import logging

from collections.abc import Iterable
from collections.abc import Iterator

from cs336_basics.io_utils import deserialize_merges
from cs336_basics.io_utils import deserialize_vocab
from cs336_basics.bpe import get_pretokens

# Create a logger specific to this file module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()

# Create a formatting layout
formatter = logging.Formatter('%(name)s - %(levelname)s - %(asctime)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to your logger
logger.addHandler(console_handler)


class Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str]|None = None):
        logger.info("Tokenizer created.")
        self.vocab = vocab
        self.vocab_rev:dict[bytes,int] = {}
        for index,bytestr in vocab.items():
            self.vocab_rev[bytestr] = index
        logger.debug("Vocab successfully loaded")

        self.merges:dict[tuple[bytes,bytes],int] = {}
        rank = 0
        for merge_pair in merges:
            self.merges[merge_pair] = rank
            rank+=1
            # assert merge not in merges
        logger.debug("Merges successfully loaded")

        if special_tokens is not None:
            for special_token in special_tokens:
                encoded =special_token.encode("utf-8")
                if encoded not in self.vocab_rev:
                    len_vocab = len(self.vocab)
                    self.vocab[len_vocab] = encoded
                    self.vocab_rev[encoded] = len_vocab
        self.special_tokens = special_tokens
        logger.debug("Special tokens are %s",self.special_tokens)

    @classmethod
    def from_files(cls,vocab_filepath:str,merges_filepath:str,special_tokens:list[str]|None=None):
        logger.info("Creating a Tokenizer")
        vocab = deserialize_vocab(vocab_filepath)
        merges = deserialize_merges(merges_filepath)
        return cls(vocab = vocab,merges = merges,special_tokens = special_tokens)

    def encode_pretoken(self,pretoken:str)->list[int]:
        b_array = list(map(int.to_bytes, pretoken.encode('utf-8')))
        # b_array
        id_list = []
        merges_complete = False
        while not merges_complete:
            b_array_len = len(b_array)
            logger.debug("Pretoken representation is %s", b_array)
            if b_array_len == 1:
                logger.debug("Only one element left")
                # id_list.append(tokenizer.vocab_rev[b_array[0]])
                merges_complete = True
            elif b_array_len == 2:
                candidate_pair = (b_array[0], b_array[1])
                if candidate_pair in self.merges:
                    # logger.debug("Found candidate pair %s for last two elements!", candidate_pair)
                    b_array = [b_array[0] + b_array[1]]
                else:
                    # logger.debug("Did not find candidate pair %s for last two elements!", {candidate_pair})
                    merges_complete = True
            else:  # 3 or greater
                new_b_array = []
                lowest_rank = 999999
                best_candidate = None
                found_index = -1
                for counter in range(0, b_array_len - 1):
                    candidate_pair = (b_array[counter], b_array[counter + 1])
                    if candidate_pair in self.merges:
                        candidate_rank = self.merges[candidate_pair]
                        if candidate_rank < lowest_rank:
                            lowest_rank = candidate_rank
                            best_candidate = candidate_pair
                            found_index = counter

                if best_candidate is not None:
                    logger.debug(f"Best candidate is %s found at index %d", best_candidate, found_index)
                    if found_index > 0:
                        new_b_array.extend(b_array[:found_index])
                    new_b_array.append(best_candidate[0] + best_candidate[1])
                    if found_index < b_array_len - 2:
                        new_b_array.extend(b_array[found_index + 2:])
                    b_array = new_b_array
                else:
                    # no merges found. abort
                    logger.debug("Could not merge for 3 or greater. Aborting search")
                    merges_complete = True
        for b in b_array:
            id_list.append(self.vocab_rev[b])
        logger.debug("b_array is %s Id list is %s", b_array, id_list)
        return id_list

    def encode(self,text:str)->list[int]:
        pretokens = get_pretokens(text,self.special_tokens)
        pretokens_encoded = []
        for pretoken in pretokens:
            if len(pretoken)>0:
                if self.special_tokens is not None and pretoken in self.special_tokens:
                    st_id = self.vocab_rev[pretoken.encode("utf-8")]
                    logger.debug("Appending special token ID %d",st_id)
                    pretokens_encoded.append(st_id)
                else:
                    id_list = self.encode_pretoken(pretoken)
                    logger.debug("Extending token id list %s", id_list)
                    pretokens_encoded.extend(id_list)
        return pretokens_encoded

    def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:
        for full_string in iterable:
            encoded_str = self.encode(full_string)
            pretokens_encoded: list[int] = encoded_str
            for encoded in pretokens_encoded:
                yield encoded


    def decode(self,ids:list[int])->str:
        if ids is None or len(ids) == 0:
            return ""
        b_array = []
        for id in ids:
            b_array.append(self.vocab[id])
        return (b"".join(b_array)).decode("utf-8", errors="replace")

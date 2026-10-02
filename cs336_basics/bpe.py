import os
import logging
from collections import Counter
import time
import regex as re
import itertools
from concurrent.futures import ProcessPoolExecutor

from cs336_basics.io_utils import find_chunk_boundaries

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


def get_pretoken_counts_worker(use_findall:bool,special_tokens: list[str], filename: str, start_end: tuple[int, int]) -> dict[str, int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counts: dict[str, int] = Counter({})
    (start, end) = start_end
    with open(filename, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
        regex_pattern = "|".join(map(re.escape, special_tokens))
        # only taking the first 'document' for prototyping. when things are working, will generalize remaining code into a function and loop over elements
        file_content_docs = re.split(regex_pattern, chunk)

        for file_content in file_content_docs:
            # logger.debug(f"regex pattern is {regex_pattern} Filtered {file_content}")
            if use_findall:
                match_list = re.findall(PAT,file_content)
                for pretoken_str in match_list:

                    counts[pretoken_str] = counts.get(pretoken_str, 0) + 1
            else:
                pretoken_iter = re.finditer(PAT, file_content)
                for pretoken in pretoken_iter:
                    pretoken_str = pretoken.group(0)
                    counts[pretoken_str] = counts.get(pretoken_str, 0) + 1
    return counts

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creationprint(.
    """



    CHECK_INVARIANTS = False
    COUNT_OVERLAPS = True
    MAX_TOKENS=vocab_size
    #with open('../tests/fixtures/tinystories_sample_5M.txt', 'r') as f:

    # special_tokens=["<|endoftext|>","<|endoftext|><|endoftext|>","GARYSPLIT"]
    special_tokens =sorted(special_tokens,key=lambda k:len(k),reverse=True)
    logger.debug(f"Special tokens are: {special_tokens}")

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # phase 1: tally counts of each pre-token into a dict where the key is a tokens
    # should be generalized to a Unicode code point

    # this variable contains the total counts on the corpus for each "word"/"pretoken"
    counts_by_pretoken = Counter({})

    # this variable contains a list of ints that represent the byte array for a token ID
    tokens_by_pretoken = {}
    # this variable maps token IDs to byte arrays
    vocab = {}

    # the merges that happened
    merges=[]

    # optimization
    incremental_counts_by_token_pair = {}

    # optimization 2
    # dict of token pairs as keys, set of pretoken strings as values
    pretokens_by_token_pair = {}

    i=0
    for j in range(256):    #logger.debug(f"i is {i} with char {chr(i)}")
        vocab[i]=i.to_bytes()
        i+=1
    for special_token in special_tokens:
        vocab[i]=special_token.encode("utf-8")
        i+=1

    logger.debug(f"Initial vocab dict {vocab}")


    # pretokens_eligible_list = []

    # a list of token ID pairs, sorted by their counts in descending order


    def load_file(filename: str | os.PathLike):
        with open(filename, 'r') as f:
            logger.debug(f"Loading file {filename}")
            file_content = f.read()
            regex_pattern = "|".join(map(re.escape,special_tokens))
            # only taking the first 'document' for prototyping. when things are working, will generalize remaining code into a function and loop over elements
            file_content_docs = re.split(regex_pattern,file_content)
            for file_content in file_content_docs:
                #logger.debug(f"regex pattern is {regex_pattern} Filtered {file_content}")
                pretoken_iter = re.finditer(PAT, file_content)
                for pretoken in pretoken_iter:
                    pretoken_str = pretoken.group(0)
                    counts_by_pretoken[pretoken_str] = counts_by_pretoken.get(pretoken_str, 0) + 1
        for pretoken in counts_by_pretoken.keys():
            tokens_by_pretoken[pretoken] = list(pretoken.encode("utf-8"))



    def load_file_parallel(use_find_all:bool,num_processes:int,infile: str | os.PathLike,special_tokens:list[str]):
        split_bytes = "|".join(map(re.escape, special_tokens)).encode("utf-8")
        # chunk_counts = Counter({})
        # token_lists = {}
        document_delimiter = "<|endoftext|>".encode("utf-8")
        with open(infile, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_processes, document_delimiter)
            start_end_list = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
            logger.debug(f"Boundaries are {start_end_list}")
            with ProcessPoolExecutor() as executor:
                # executor.map applies the function to each element in parallel
                parallel_list = list(executor.map(get_pretoken_counts_worker, itertools.repeat(use_find_all),itertools.repeat(special_tokens), itertools.repeat(infile),start_end_list))

                for e in parallel_list:
                    counts_by_pretoken.update(e)
                    # chunk_counts += e
                for pretoken in counts_by_pretoken.keys():
                    tokens_by_pretoken[pretoken] = list(pretoken.encode("utf-8"))




    def get_counts_by_token_pair(tokens_by_pretoken:dict[str,list[int]], counts_by_pretoken:dict[str,int])->dict[tuple[int,int],int]:
        counts_by_tokenpair = {}
        for pretoken in tokens_by_pretoken.keys():
            pretoken_freq = counts_by_pretoken[pretoken]
            token_list = tokens_by_pretoken[pretoken]#token_pair_frequency = {}
            for (token0,token1) in itertools.pairwise(token_list):
                counts_by_tokenpair[(token0,token1)] = counts_by_tokenpair.get((token0,token1),0) + pretoken_freq
                pretokens_by_token_pair[(token0,token1)] =pretokens_by_token_pair.get((token0,token1),set())
                pretokens_by_token_pair[(token0, token1)].add(pretoken)

        return counts_by_tokenpair


    def get_top_token_pair()->tuple[tuple[int,int],int]:
        # this data structure should contain the key as the tuple of token0 and token 1 and the value as a tuple of the counts, and a list of pre tokens it belong to
        counts_by_tokenpair = incremental_counts_by_token_pair
        best_count = -1
        best_token_pair = (-1,-1)
        best_byte_pair = None
        for tokenpair,freq in counts_by_tokenpair.items():
            if freq > best_count:
                best_count = freq
                best_token_pair = tokenpair
                # logger.debug(f"fetching vocab index {tokenpair[0]} and {tokenpair[1]}")
                best_byte_pair = (vocab[tokenpair[0]],vocab[tokenpair[1]])
            elif freq == best_count and  (vocab[tokenpair[0]],vocab[tokenpair[1]]) > best_byte_pair:
                best_count = freq
                best_token_pair = tokenpair
                best_byte_pair = (vocab[tokenpair[0]],vocab[tokenpair[1]])
        candidate = (best_token_pair,best_count)
        # logger.debug(f"Candidate {candidate}")
        return candidate
        # return (max_key,max_val)

    def apply_top_merge(token_pair:tuple[int,int]):
        def apply_decrement(t0:int,t1:int,delta_index:int,deltas_set:dict[tuple[int,int],int],pretoken_frequency:int,log_mesg:str):
            if ((t0, t1), delta_index) not in deltas_set:
                logger.debug(f"Decrementing {((t0, t1), delta_index)} from {incremental_counts_by_token_pair.get((t0, t1),0)} by {pretoken_frequency}")
                incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),
                                                                                                  0) - pretoken_frequency
                if incremental_counts_by_token_pair[(t0, t1)] == 0:
                    logger.debug(f"HOORAY {(t0,t1)} WENT TO ZERO")
                    del incremental_counts_by_token_pair[(t0, t1)]
                    if (t0,t1) in pretokens_by_token_pair:
                        del pretokens_by_token_pair[(t0, t1)]
            else:
                logger.debug(log_mesg)
            deltas_set.add(((t0, t1), delta_index))

        def apply_increment(t0:int,t1:int,delta_index:int,deltas_set:dict[tuple[int,int],int],log_mesg:str,pretoken_frequency:int,pretoken:str):
            if ((t0, t1), delta_index) not in deltas_set:
                logger.debug(log_mesg)
                logger.debug(f"Incrementing {((t0, t1), delta_index)} from {incremental_counts_by_token_pair.get((t0, t1), 0)} by {pretoken_frequency}")
                incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),
                                                                                                  0) + pretoken_frequency
                pretokens_by_token_pair[(t0,t1)] =pretokens_by_token_pair.get((t0,t1),set())
                pretokens_by_token_pair[(t0, t1)].add(pretoken)
            deltas_set.add(((t0, t1), delta_index))


        new_token_id = len(vocab)
        # the byte string concatenation
        vocab[new_token_id] = vocab[token_pair[0]] + vocab[token_pair[1]]
        # pretokens_eligible = 0

        pretoken_set = pretokens_by_token_pair[(token_pair[0],token_pair[1])]
        for pretoken in pretoken_set:
            token_ints = tokens_by_pretoken[pretoken]
            pretoken_frequency = counts_by_pretoken[pretoken]
            # logger.debug(f"Pretoken {pretoken} with frequency {pretoken_frequency}")
            # token_int_len = len(token_ints)
            insertion_indices = []

            # first sweep through and gather the indices that would need insertions of the new token id
            counter = 0
            for (token_int0, token_int1) in itertools.pairwise(token_ints):
                if token_int0 == token_pair[0] and token_int1 == token_pair[1]:
                    insertion_indices.append(counter)
                counter += 1
            token_int_len = counter + 1
            insertion_indices_len = len(insertion_indices)
            if insertion_indices_len == 0:
                continue
            # else:
            #     pretokens_eligible+=1
            logger.debug(f"Token list for pretoken {pretoken} with freq {pretoken_frequency} is {token_ints}")

            delta = 3
            for counter in range(insertion_indices_len):
                if counter > 0:
                    delta = insertion_indices[counter] - insertion_indices[counter-1]
                    if delta == 1:
                        logger.debug(f"edge case of overlapping: {token_ints}")
                        break
                        # assert (False)
                    elif delta == 2:
                        logger.debug(f"edge case of neighboring pairs: {token_ints}")
                        break
                        # assert(False)

            if delta ==1:
                insertion_indices_pruned = []
                insertion_indices_pruned.append(insertion_indices[0])

                anchor_value = insertion_indices[0]
                anchor_idx = 0
                for walker in range( insertion_indices_len-1):
                    logger.debug(f"Looking from {anchor_idx+1} to {insertion_indices_len}")
                    target_idx = walker+1
                    while target_idx< insertion_indices_len:
                        logger.debug(f"Comparing {insertion_indices[anchor_idx]} at position {anchor_idx} to {insertion_indices[target_idx]} at position {(target_idx)}")
                        if (insertion_indices[target_idx] - insertion_indices[anchor_idx]) > 1:
                            insertion_indices_pruned.append(insertion_indices[target_idx])
                            anchor_idx = target_idx
                        target_idx+=1
                logger.debug(f"Insertion indices_len {insertion_indices_len} counter {counter} old insertion indices {insertion_indices} new insertion indices {insertion_indices_pruned}")
                insertion_indices = insertion_indices_pruned
                insertion_indices_len = len(insertion_indices_pruned)

            # decrementing pairs straddling only of the elements of token_pair
            # don't double count for decrements
            # this set is ((token0,token1),index)
            deltas_set:dict[tuple[int,int],int] = set()
            for insertion_index in insertion_indices:
                # handle case where at very left
                if insertion_index == 0:
                    if token_int_len > 2:
                        t0, t1 =  token_ints[insertion_index + 1], token_ints[insertion_index + 2]
                        delta_index =insertion_index + 1
                        log_mesg = f"edge case {delta_index} with right neighbor was already decremented"
                        apply_decrement(t0, t1, delta_index, deltas_set, pretoken_frequency, log_mesg)
                elif insertion_index == (token_int_len - 2):
                    # at very right
                    if token_int_len > 2:
                        t0, t1 = token_ints[insertion_index -1], token_ints[insertion_index ]
                        delta_index = insertion_index - 1
                        log_mesg = f"edge case {delta_index} with left neighbor was already decremented"
                        apply_decrement(t0, t1, delta_index, deltas_set, pretoken_frequency, log_mesg)
                # in the moddle
                else:
                    if token_int_len > 3:
                        # something is flanking on left
                        t0, t1 = token_ints[insertion_index - 1], token_ints[insertion_index]
                        delta_index = insertion_index - 1
                        log_mesg = f"edge case {delta_index} with flanking neighbors was already decremented"
                        apply_decrement(t0, t1, delta_index, deltas_set, pretoken_frequency, log_mesg)
                        # something is flanking on right
                        t0, t1 = token_ints[insertion_index +1], token_ints[insertion_index+2]
                        delta_index = insertion_index + 1
                        log_mesg = f"edge case {delta_index} with flanking neighbors was already decremented"
                        apply_decrement(t0, t1, delta_index, deltas_set, pretoken_frequency, log_mesg)
                logger.debug(f"Decrement set {deltas_set}")

            new_list2 = token_ints.copy()
            new_list2_len = token_int_len
            for insertion_index in reversed(insertion_indices):
                del new_list2[insertion_index+1]
                del new_list2[insertion_index]
                new_list2.insert(insertion_index,new_token_id)
                new_list2_len-=1
            tokens_by_pretoken[pretoken] = new_list2
            logger.debug(f"New token list for pretoken {pretoken} with freq {pretoken_frequency} is {new_list2} New token is {new_token_id}")
            # incrementing pairs straddling only of the elements of token_pair
            token_int_len = new_list2_len
            deltas_set.clear()
            for counter in range(token_int_len):
                if new_list2[counter] == new_token_id:
                    # handle case where at very left
                    if counter == 0:
                        if token_int_len > 1:
                            t0, t1 =  new_list2[counter], new_list2[counter + 1]
                            delta_index = counter
                            log_mesg = f"At very left, incrementing right neighbor {(t0, t1)} at by {pretoken_frequency}"
                            apply_increment(t0,t1,delta_index,deltas_set,log_mesg,pretoken_frequency,pretoken)
                    elif counter == (token_int_len - 1):
                        # at very right
                        if token_int_len > 1:
                            t0, t1 = new_list2[counter-1], new_list2[counter]
                            delta_index = counter-1
                            log_mesg = f"At very right, incrementing left neighbor {(t0, t1)} at by {pretoken_frequency}"
                            apply_increment(t0,t1,delta_index,deltas_set,log_mesg,pretoken_frequency,pretoken)
                            if ((t0, t1), (counter - 1)) not in deltas_set:
                                # logger.debug(f"At very right, incrementing left neighbor {(t0, t1)} at by {pretoken_frequency}")
                                incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) + pretoken_frequency
                            deltas_set.add(((t0, t1), counter-1))
                    # in the moddle
                    else:
                        if token_int_len > 2:
                            # something is flanking on left
                            t0, t1 = new_list2[counter - 1], new_list2[counter]
                            delta_index = counter-1
                            log_mesg = f"Incrementing left neighbor {(t0, t1)} at index {counter - 1} by {pretoken_frequency}"
                            apply_increment(t0,t1,delta_index,deltas_set,log_mesg,pretoken_frequency,pretoken)
                            # something is flanking on right
                            t0, t1 = new_list2[counter ], new_list2[counter+1]
                            delta_index = counter
                            log_mesg = f"Incrementing right neighbor {(t0, t1)} at index {counter} by {pretoken_frequency}"
                            apply_increment(t0,t1,delta_index,deltas_set,log_mesg,pretoken_frequency,pretoken)
            # decrementing all obsoleted pairs
            logger.debug(f"Insertion indices len is {insertion_indices_len} and token pair is {(token_pair[0], token_pair[1])}")
            if insertion_indices_len> 0 and (token_pair[0], token_pair[1]) in incremental_counts_by_token_pair :
                logger.debug(f"Decrementing pair {token_pair} from {incremental_counts_by_token_pair[(token_pair[0], token_pair[1])]} by {insertion_indices_len * pretoken_frequency}")
                incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] -= insertion_indices_len * pretoken_frequency
                assert (incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] >= 0)
                if incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] == 0:
                    # logger.debug(f"HOORAY {(token_pair[0], token_pair[1])} WENT TO ZERO")
                    del incremental_counts_by_token_pair[(token_pair[0], token_pair[1])]
                    if (token_pair[0],token_pair[1]) in pretokens_by_token_pair:
                        del pretokens_by_token_pair[(token_pair[0],token_pair[1])]


            if CHECK_INVARIANTS:
                logger.debug(f"Checking incremental vs gold standard")
                counts_by_token_pair_oracle = get_counts_by_token_pair(tokens_by_pretoken, counts_by_pretoken)
                if incremental_counts_by_token_pair != counts_by_token_pair_oracle:
                    debug_dicts(incremental_counts_by_token_pair, counts_by_token_pair_oracle)
                assert (counts_by_token_pair_oracle == incremental_counts_by_token_pair)



    def debug_dicts(incremental, truth):
        # first loop through incremental
        for k in incremental:
            if k not in truth:
                logger.debug(f"Incremental has spurious pair {k} with values {incremental[k]}")
            elif incremental[k] != truth[k]:
                logger.debug(f"Discrepancy: {k} incremental value is {incremental[k]} and truth is {truth[k]}")
        for k in truth:
            if k not in incremental:
                logger.debug(f"Incremental is missing key {k} which should have values {truth[k]}")


    def apply_merges()->tuple[dict[int,bytes],list[tuple[int,int]]]:
        i = 0
        status_interval = 500
        start_time = time.perf_counter()
        while True:
            (token_pair,freq)= get_top_token_pair()
            if freq>0:
                merges.append((vocab[token_pair[0]],vocab[token_pair[1]]))
                apply_top_merge(token_pair)

            else:
                logger.debug(f"Nothing was merged. Aborting")
                break
            if len(vocab)>=MAX_TOKENS :
                logger.debug(f"Ending with vocab length {len(vocab)} . Aborting")
                break
            if i%status_interval==0:
                end_time = time.perf_counter()
                logger.info(f"{i} merges completed with time {end_time-start_time}.")
                start_time = end_time

            i+=1

    # start_time = time.perf_counter()
    #load_file(input_path)

    # end_time = time.perf_counter()
    # logger.info(f"Old load is {(end_time-start_time)}")
    start_time = time.perf_counter()
    workers = 4
    # counts_by_pretoken.clear()
    # load_file_parallel(False,workers,input_path,special_tokens)
    # oldcounts = counts_by_pretoken.copy()
    # #print(oldcounts)
    # counts_by_pretoken.clear()
    load_file_parallel(True, workers, input_path, special_tokens)
    # newcounts = counts_by_pretoken.copy()
    #print(newcounts)
    # assert(newcounts==oldcounts)
    end_time = time.perf_counter()
    logger.info(f"Parallel load time at {workers} workers is {(end_time - start_time)}")

    incremental_counts_by_token_pair = get_counts_by_token_pair(tokens_by_pretoken,counts_by_pretoken)
    end_time2 = time.perf_counter()
    logger.info(f"Initializing invariants table time is {(end_time2 - end_time)}")
    logger.debug(f"Baseline: {incremental_counts_by_token_pair}")
    apply_merges()
    end_time3 = time.perf_counter()
    logger.info(f"Applying merges time is {(end_time3 - end_time)}")
    # if CHECK_INVARIANTS:
    #     logger.debug(f"Checking incremental vs gold standard")
    #     counts_by_token_pair_oracle = get_counts_by_token_pair(tokens_by_pretoken,counts_by_pretoken)
    #     if incremental_counts_by_token_pair != counts_by_token_pair_oracle:
    #         debug_dicts(incremental_counts_by_token_pair, counts_by_token_pair_oracle)
        # assert(counts_by_token_pair_oracle==incremental_counts_by_token_pair)
    return (vocab,merges)

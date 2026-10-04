import os
import logging
from collections import Counter
import time
import regex as re
import itertools
import heapq
from concurrent.futures import ProcessPoolExecutor

from cs336_basics.io_utils import find_chunk_boundaries

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

class HeapElement:
    __slots__ = ['token_pair','counts','bytes_pair','stale']
    def __init__(self,token_pair:tuple[int,int],counts:int,bytes_pair:tuple[bytes,bytes]):
        self.token_pair = token_pair
        self.counts = counts
        self.bytes_pair = bytes_pair
        self.stale = False
    def __lt__(self,other):
        if self.counts!=other.counts:
            return self.counts>other.counts
        return self.bytes_pair > other.bytes_pair
    def __repr__(self):
        return f"Token pair  {self.token_pair} counts {self.counts} bytes pair {self.bytes_pair} is stale? {self.stale}"

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
            # logger.debug("regex pattern is {regex_pattern} Filtered {file_content}")
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
    workers:int = 1,
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
    CHECK_PRUNE_LIST = False
    CHECK_HEAP = False

    MAX_TOKENS=vocab_size
    #with open('../tests/fixtures/tinystories_sample_5M.txt', 'r') as f:

    # special_tokens=["<|endoftext|>","<|endoftext|><|endoftext|>","GARYSPLIT"]
    special_tokens =sorted(special_tokens,key=lambda k:len(k),reverse=True)
    logger.debug("Special tokens are: %s",special_tokens)

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

    #optimization 3 using heap to improve get_top_pair
    # the lookup will make a heap element stale if necessary
    heap_element_lookup = {} # key is (token0,token) value is HeapElement
    token_pair_heap = []

    i=0
    for j in range(256):    #logger.debug("i is {i} with char {chr(i)}")
        vocab[i]=i.to_bytes()
        i+=1
    for special_token in special_tokens:
        vocab[i]=special_token.encode("utf-8")
        i+=1

    logger.debug("Initial vocab dict %s",vocab)


    # pretokens_eligible_list = []

    # a list of token ID pairs, sorted by their counts in descending order


    def load_file(filename: str | os.PathLike):
        with open(filename, 'r') as f:
            logger.debug("Loading file %",filename)
            file_content = f.read()
            regex_pattern = "|".join(map(re.escape,special_tokens))
            # only taking the first 'document' for prototyping. when things are working, will generalize remaining code into a function and loop over elements
            file_content_docs = re.split(regex_pattern,file_content)
            for file_content in file_content_docs:
                #logger.debug("regex pattern is {regex_pattern} Filtered {file_content}")
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
            logger.debug("Boundaries are %s",start_end_list)
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
                counts = counts_by_tokenpair.get((token0,token1),0) + pretoken_freq
                counts_by_tokenpair[(token0,token1)] = counts
                pretokens_by_token_pair[(token0,token1)] =pretokens_by_token_pair.get((token0,token1),set())
                pretokens_by_token_pair[(token0, token1)].add(pretoken)

        if not token_pair_heap:
            for (token_pair,counts) in counts_by_tokenpair.items():
                add_to_heap(token_pair, counts)
        return counts_by_tokenpair


    def get_top_token_pair()->tuple[tuple[int,int],int]:
        if CHECK_HEAP:
            # this data structure should contain the key as the tuple of token0 and token 1 and the value as a tuple of the counts, and a list of pre tokens it belong to
            counts_by_tokenpair = incremental_counts_by_token_pair
            best_count = -1
            best_token_pair = (-1,-1)
            best_byte_pair = None
            for tokenpair,freq in counts_by_tokenpair.items():
                if freq > best_count:
                    best_count = freq
                    best_token_pair = tokenpair
                    # logger.debug("fetching vocab index {tokenpair[0]} and {tokenpair[1]}")
                    best_byte_pair = (vocab[tokenpair[0]],vocab[tokenpair[1]])
                elif freq == best_count and  (vocab[tokenpair[0]],vocab[tokenpair[1]]) > best_byte_pair:
                    best_count = freq
                    best_token_pair = tokenpair
                    best_byte_pair = (vocab[tokenpair[0]],vocab[tokenpair[1]])
            candidate = (best_token_pair,best_count)

        last_stale = True
        candidate2 = None
        while last_stale:
            if not token_pair_heap:
                logger.warning("Token pair heap is empty!")
                break
            else:
                candidate2 = heapq.heappop(token_pair_heap)
                last_stale = candidate2.stale
                #print(popped)
        if CHECK_HEAP:
            logger.debug(f"Candidate {candidate} Candidate2 {candidate2}")
            assert(candidate==(candidate2.token_pair,candidate2.counts))
        return (candidate2.token_pair,candidate2.counts)
        # return (max_key,max_val)

    def add_to_heap(token_pair:tuple[int,int],new_count):
        # create a new reference
        logger.debug("Adding to heap %s with new count %s",token_pair,new_count)
        new_heap_element = HeapElement(token_pair, new_count, (vocab[token_pair[0]], vocab[token_pair[1]]))
        # push this new reference
        heapq.heappush(token_pair_heap, new_heap_element)
        # overwrite the reference in the lookup
        heap_element_lookup[token_pair] = new_heap_element

    # doesn't actually remove from heap but marks it as stale
    def remove_from_heap(token_pair:tuple[int,int]):
        if token_pair in heap_element_lookup:
            # get the reference to the element in the heap
            stale_heap_element = heap_element_lookup[token_pair]

            # assert(stale_heap_element.stale == heap_element_lookup[token_pair].stale)
            # mark as stale
            stale_heap_element.stale = True
            logger.debug("Marked as stale %s", stale_heap_element)


    def update_heap(token_pair:tuple[int,int],new_count):
        remove_from_heap(token_pair)
        add_to_heap(token_pair,new_count)


    def apply_top_merge(token_pair:tuple[int,int]):
        pairs_updated = set()

        def apply_decrement(t0:int,t1:int,delta_index:int,deltas_set:set[tuple[int,int],int],pretoken_frequency:int,log_mesg:str):
            if ((t0, t1), delta_index) not in deltas_set:
                logger.debug("Decrementing %s from %d by %d",((t0, t1), delta_index),incremental_counts_by_token_pair.get((t0, t1),0),pretoken_frequency)
                new_count = incremental_counts_by_token_pair.get((t0, t1),0) - pretoken_frequency
                incremental_counts_by_token_pair[(t0, t1)] = new_count
                # update_heap((t0,t1),new_count)
                pairs_updated.add((t0,t1))
                if incremental_counts_by_token_pair[(t0, t1)] == 0:
                    logger.debug("Token pair %s went to zero so removing",(t0,t1))
                    del incremental_counts_by_token_pair[(t0, t1)]
                    remove_from_heap((t0,t1))
                    if (t0,t1) in pretokens_by_token_pair:
                        del pretokens_by_token_pair[(t0, t1)]
            else:
                logger.debug(log_mesg)
            deltas_set.add(((t0, t1), delta_index))

        def apply_increment(t0:int,t1:int,log_mesg:str,pretoken_frequency:int,pretoken:str,delta_index:int,deltas_set:set[tuple[int,int],int]):
            if ((t0, t1), delta_index) not in deltas_set:
                logger.debug("Incrementing %s with number %d by %d",(t0, t1),incremental_counts_by_token_pair.get((t0, t1), 0),pretoken_frequency)
                new_count = incremental_counts_by_token_pair.get((t0, t1),0) + pretoken_frequency
                incremental_counts_by_token_pair[(t0, t1)] = new_count
                # update_heap((t0,t1),new_count)
                pairs_updated.add((t0, t1))
                pretokens_by_token_pair[(t0,t1)] = pretokens_by_token_pair.get((t0,t1),set())
                pretokens_by_token_pair[(t0,t1)].add(pretoken)
            else:
                logger.debug(log_mesg)
            deltas_set.add(((t0, t1), delta_index))



        new_token_id = len(vocab)
        # the byte string concatenation
        vocab[new_token_id] = vocab[token_pair[0]] + vocab[token_pair[1]]
        # pretokens_eligible = 0

        pretoken_set = pretokens_by_token_pair[(token_pair[0],token_pair[1])]
        assert(len(pretoken_set)>0)
        for pretoken in pretoken_set:
            token_ints = tokens_by_pretoken[pretoken]
            pretoken_frequency = counts_by_pretoken[pretoken]

            logger.debug("Pretoken %s with frequency %d token_ints %s",pretoken,pretoken_frequency,token_ints)
            logger.debug("Looking for token pair %d %d",token_pair[0],token_pair[1])
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
            logger.debug("Token list for pretoken %s with freq %d is %s",pretoken,pretoken_frequency,token_ints)

            delta = 3

            for counter in range(insertion_indices_len):
                if counter > 0:
                    delta = insertion_indices[counter] - insertion_indices[counter-1]
                    if delta == 1:
                        logger.debug("edge case of overlapping: %s",token_ints)
                        break
                        # assert (False)
                    elif delta == 2:
                        logger.debug("edge case of neighboring pairs: %s",token_ints)
                        break
                        # assert(False)
            if delta ==1:
                insertion_indices_testing = insertion_indices.copy()
                if CHECK_PRUNE_LIST:
                    insertion_indices_pruned = []
                    insertion_indices_pruned.append(insertion_indices[0])

                    anchor_value = insertion_indices[0]
                    anchor_idx = 0
                    for walker in range( insertion_indices_len-1):
                        logger.debug("Looking from %d to %d",anchor_idx+1,insertion_indices_len)
                        target_idx = walker+1
                        while target_idx< insertion_indices_len:
                            if (insertion_indices[target_idx] - insertion_indices[anchor_idx]) > 1:
                                insertion_indices_pruned.append(insertion_indices[target_idx])
                                anchor_idx = target_idx
                            target_idx+=1
                    logger.debug("Insertion indices_len %d counter %d old insertion indices %s new insertion indices %s",insertion_indices_len,counter,insertion_indices,insertion_indices_pruned)
                    insertion_indices = insertion_indices_pruned
                    insertion_indices_len = len(insertion_indices_pruned)

                prune_list = []
                pruned = False
                for repeat_walker in range(1, len(insertion_indices_testing)):
                    logger.debug("Repeat walker at %",repeat_walker)
                    if not pruned and (insertion_indices_testing[repeat_walker] - insertion_indices_testing[repeat_walker - 1] == 1):
                        prune_list.append(repeat_walker)
                        pruned = True
                    else:
                        pruned = False
                for prune_index in reversed(prune_list):
                    del insertion_indices_testing[prune_index]
                if CHECK_PRUNE_LIST:
                    assert(insertion_indices_pruned==insertion_indices_testing)
                insertion_indices = insertion_indices_testing
                insertion_indices_len = len(insertion_indices_testing)


            # decrementing pairs straddling only of the elements of token_pair
            # don't double count for decrements
            # this set is ((token0,token1),index)
            logger.debug("Original insertion indices: %s",insertion_indices)
            deltas_set:set[tuple[int,int],int] = set()
            for insertion_index in insertion_indices:
                # handle case where at very left
                if insertion_index == 0:
                    if token_int_len > 2:
                        t0, t1 =  token_ints[insertion_index + 1], token_ints[insertion_index + 2]
                        delta_index =insertion_index + 1
                        log_mesg = f"edge case at index {delta_index} with right neighbor was already decremented"
                        apply_decrement(t0, t1, delta_index, deltas_set, pretoken_frequency, log_mesg)
                elif insertion_index == (token_int_len - 2):
                    # at very right
                    if token_int_len > 2:
                        t0, t1 = token_ints[insertion_index -1], token_ints[insertion_index ]
                        delta_index = insertion_index - 1
                        log_mesg = f"edge case at index {delta_index} with left neighbor was already decremented"
                        apply_decrement(t0, t1, delta_index, deltas_set, pretoken_frequency, log_mesg)
                # in the moddle
                else:
                    if token_int_len > 3:
                        # something is flanking on left
                        t0, t1 = token_ints[insertion_index - 1], token_ints[insertion_index]
                        delta_index = insertion_index - 1
                        log_mesg = f"edge case at index {delta_index} with flanking neighbors was already decremented"
                        apply_decrement(t0, t1, delta_index, deltas_set, pretoken_frequency, log_mesg)
                        # something is flanking on right
                        t0, t1 = token_ints[insertion_index +1], token_ints[insertion_index+2]
                        delta_index = insertion_index + 1
                        log_mesg = f"edge case at index {delta_index} with flanking neighbors was already decremented"
                        apply_decrement(t0, t1, delta_index, deltas_set, pretoken_frequency, log_mesg)
                # logger.debug("Decrement set {deltas_set}")

            new_list2 = token_ints.copy()
            new_list2_len = token_int_len
            for insertion_index in reversed(insertion_indices):
                del new_list2[insertion_index+1]
                del new_list2[insertion_index]
                new_list2.insert(insertion_index,new_token_id)
                new_list2_len-=1
            tokens_by_pretoken[pretoken] = new_list2
            logger.debug("New token list for pretoken %s with freq %d is %s New token is %d",pretoken,pretoken_frequency,new_list2,new_token_id)
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
                            log_mesg = f"edge case at index {delta_index} with right neighbor was already incremented"
                            apply_increment(t0,t1,log_mesg,pretoken_frequency,pretoken,delta_index,deltas_set)
                    elif counter == (token_int_len - 1):
                        # at very right
                        if token_int_len > 1:
                            t0, t1 = new_list2[counter-1], new_list2[counter]
                            delta_index = counter-1
                            log_mesg = f"edge case at index {delta_index} with left neighbor was already incremented"
                            apply_increment(t0,t1,log_mesg,pretoken_frequency,pretoken,delta_index,deltas_set)
                            # if ((t0, t1), (counter - 1)) not in deltas_set:
                            #     # logger.debug("At very right, incrementing left neighbor {(t0, t1)} at by {pretoken_frequency}")
                            #     incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) + pretoken_frequency
                            # deltas_set.add(((t0, t1), counter-1))
                    # in the moddle
                    else:
                        if token_int_len > 2:
                            # something is flanking on left
                            t0, t1 = new_list2[counter - 1], new_list2[counter]
                            delta_index = counter-1
                            log_mesg = f"edge case at index {delta_index} with flanking neighbors was already incremented"
                            apply_increment(t0,t1,log_mesg,pretoken_frequency,pretoken,delta_index,deltas_set)
                            # something is flanking on right
                            t0, t1 = new_list2[counter ], new_list2[counter+1]
                            delta_index = counter
                            log_mesg = f"edge case at index {delta_index} with flanking neighbors was already incremented"
                            apply_increment(t0,t1,log_mesg,pretoken_frequency,pretoken,delta_index,deltas_set)
            # decrementing all obsoleted pairs
            logger.debug("Insertion indices len is %d and token pair is %s",insertion_indices_len,(token_pair[0], token_pair[1]))
            if insertion_indices_len> 0 and (token_pair[0], token_pair[1]) in incremental_counts_by_token_pair :
                count_to_remove = insertion_indices_len * pretoken_frequency
                logger.debug("Decrementing pair %s from %d by %d",token_pair,incremental_counts_by_token_pair[(token_pair[0], token_pair[1])],count_to_remove)
                incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] -= count_to_remove
                # update_heap((token_pair[0],token_pair[1]),incremental_counts_by_token_pair[(token_pair[0], token_pair[1])])
                pairs_updated.add((token_pair[0],token_pair[1]))
                assert (incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] >= 0)
                if incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] == 0:
                    # logger.debug("HOORAY {(token_pair[0], token_pair[1])} WENT TO ZERO")
                    del incremental_counts_by_token_pair[(token_pair[0], token_pair[1])]
                    # remove_from_heap((token_pair[0],token_pair[1]))
                    if (token_pair[0],token_pair[1]) in pretokens_by_token_pair:
                        del pretokens_by_token_pair[(token_pair[0],token_pair[1])]

        # update the heap only once!
        logger.debug("Number of pairs to update is %d",len(pairs_updated))
        for pair_updated in pairs_updated:
            if pair_updated in incremental_counts_by_token_pair:
                new_count = incremental_counts_by_token_pair[pair_updated]
                update_heap(pair_updated, new_count)
            else:
                remove_from_heap(pair_updated)





            # if CHECK_INVARIANTS:
            #     logger.debug("Checking incremental vs gold standard")
            #     counts_by_token_pair_oracle = get_counts_by_token_pair(tokens_by_pretoken, counts_by_pretoken)
            #     if incremental_counts_by_token_pair != counts_by_token_pair_oracle:
            #         debug_dicts(incremental_counts_by_token_pair, counts_by_token_pair_oracle)
            #     assert (counts_by_token_pair_oracle == incremental_counts_by_token_pair)



    def debug_dicts(incremental, truth):
        # first loop through incremental
        for k in incremental:
            if k not in truth:
                logger.debug("Incremental has spurious pair %s with value %d",k,incremental[k])
            elif incremental[k] != truth[k]:
                logger.debug("Discrepancy: %s incremental value is %d and truth is %d",k,incremental[k],truth[k])
        for k in truth:
            if k not in incremental:
                logger.debug("Incremental is missing key %s which should have values %d",k,truth[k])


    def apply_merges()->tuple[dict[int,bytes],list[tuple[int,int]]]:
        # global variable iteration to enable shortcut to start debugging
        counter = 0
        status_interval = 500
        start_time = time.perf_counter()
        while True:
            logger.debug("Iteration %d",counter)
            (token_pair,freq)= get_top_token_pair()
            if freq>0:
                merges.append((vocab[token_pair[0]],vocab[token_pair[1]]))
                apply_top_merge(token_pair)
                if CHECK_INVARIANTS:
                    logger.debug("Checking incremental vs gold standard")
                    counts_by_token_pair_oracle = get_counts_by_token_pair(tokens_by_pretoken, counts_by_pretoken)
                    if incremental_counts_by_token_pair != counts_by_token_pair_oracle:
                        debug_dicts(incremental_counts_by_token_pair, counts_by_token_pair_oracle)
                    assert (counts_by_token_pair_oracle == incremental_counts_by_token_pair)

            else:
                logger.warning("Nothing was merged. Aborting")
                break
            if len(vocab)>=MAX_TOKENS :
                logger.warning("Ending with vocab length %d. Aborting",len(vocab))
                break
            if counter%status_interval==0:
                end_time = time.perf_counter()
                logger.info("%d merges completed with time %.2f.",counter, (end_time-start_time))
                start_time = end_time

            counter+=1

    # start_time = time.perf_counter()
    #load_file(input_path)

    # end_time = time.perf_counter()
    # logger.info(f"Old load is {(end_time-start_time)}")
    start_time = time.perf_counter()
    #workers = 4
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
    logger.debug("Baseline: {incremental_counts_by_token_pair}")
    apply_merges()
    end_time3 = time.perf_counter()
    logger.info(f"Applying merges time is {(end_time3 - end_time)}")
    if CHECK_INVARIANTS:
        logger.debug("Checking incremental vs gold standard")
        counts_by_token_pair_oracle = get_counts_by_token_pair(tokens_by_pretoken,counts_by_pretoken)
        if incremental_counts_by_token_pair != counts_by_token_pair_oracle:
            debug_dicts(incremental_counts_by_token_pair, counts_by_token_pair_oracle)
        assert(counts_by_token_pair_oracle==incremental_counts_by_token_pair)
    return (vocab,merges)

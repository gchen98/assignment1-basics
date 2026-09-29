import os
import logging

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
    import regex as re
    import os
    import itertools

    #DEBUG_TOKEN_INTS = [32, 49, 48, 48, 48, 48]
    DEBUG_TOKEN_INTS = []
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
    counts_by_pretoken = {}

    # this variable contains a list of ints that represent the byte array for a token ID
    tokens_by_pretoken = {}
    # this variable maps token IDs to byte arrays
    vocab = {}

    # the merges that happened
    merges=[]

    # optimization
    incremental_counts_by_token_pair = {}

    # optimization 2
    # pretokens_by_token_pair = {}

    i=0
    for j in range(256):    #logger.debug(f"i is {i} with char {chr(i)}")
        vocab[i]=i.to_bytes()
        i+=1
    for special_token in special_tokens:
        vocab[i]=special_token.encode("utf-8")
        i+=1

    logger.debug(f"Initial vocab dict {vocab}")

    # a list of token ID pairs, sorted by their counts in descending order

    # sorted_counts = {}
    # logger.debug(f"Sorted counts initialized length {len(sorted_counts)}")


    def load_file(filename):
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


        #vocab_set.add('<|endoftext|>')
        #vocab_list = list(vocab_set)
        #vocab_list = [for item in vocab_set]
        # logger.debug(f"Vocab dict {vocab}")
        # logger.debug(f"counts_by_pretoken {counts_by_pretoken}")
        for pretoken in counts_by_pretoken.keys():
            tokens_by_pretoken[pretoken] = list(pretoken.encode("utf-8"))
        # logger.debug(f"tokens_by_pretoken {tokens_by_pretoken}")

    def get_counts_by_token_pair(tokens_by_pretoken, counts_by_pretoken):
        counts_by_tokenpair = {}
        for pretoken in tokens_by_pretoken.keys():
            pretoken_freq = counts_by_pretoken[pretoken]
            # logger.debug(f" pretoken: {pretoken} freq:{pretoken_freq}")
            token_list = tokens_by_pretoken[pretoken]#token_pair_frequency = {}
            # logger.debug(f"last merge {last_merge} token_list {token_list}")
            # if last_merge is None:
                # logger.debug(f"Exhaustive count of all pairs")
                # last_pair = None
            for (token0,token1) in itertools.pairwise(token_list):
                # pretokens_by_token_pair[(token0,token1)] = pretokens_by_token_pair.get((token0,token1),set())
                # pretokens_by_token_pair[(token0, token1)].add(pretoken)
            # for (token0,token1) in zip(token_list,token_list[1:]):
                # if COUNT_OVERLAPS:
                #sorted_counts[(token0,token1)] = sorted_counts.get((token0,token1),0) + pretoken_freq
                # logger.debug(f"Token {token0} {token1}")
                counts_by_tokenpair[(token0,token1)] = counts_by_tokenpair.get((token0,token1),0) + pretoken_freq
                    # else:
                    #     if (last_pair is not None and last_pair!=(token0,token1)):

                    # last_pair = (token0,token1)
        return counts_by_tokenpair


    def get_top_token_pair(last_merge):

        # this data structure should contain the key as the tuple of token0 and token 1 and the value as a tuple of the counts, and a list of pre tokens it belong to
        counts_by_tokenpair = None
        if CHECK_INVARIANTS:
            logger.debug("Launching slow version to compare fast and slow versions")
            counts_by_tokenpair = get_counts_by_token_pair(tokens_by_pretoken,counts_by_pretoken)
        else:
            logger.debug("Launching express version to compare fast and slow versions")
            counts_by_tokenpair = incremental_counts_by_token_pair


        best_count = -1
        best_token_pair = None
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

            # else:
            #     if last_merge in token_list:
            #         logger.debug(f"Incremental logic on {token_list}")
            #         for (token0,token1) in zip(token_list,token_list[1:]):
            #             #token_pair_frequency[(token0,token1)] = token_pair_frequency.get((token0,token1), 0) + pretoken_freq
            #             if token0==last_merge or token1==last_merge:
            #                 # logger.debug(f"Incremental logic between {token0} and {token1}")
            #                 # if not COUNT_OVERLAPS and last_pair is not None and last_pair==(token0,token1):
            #                 #         logger.debug(f"token pair {token0,token1}")
            #
            #
            #                 sorted_counts[(token0,token1)] = sorted_counts.get((token0,token1),0) + pretoken_freq

                        #sorted_counts[(token0,token1)]  = token_pair_frequency.get((token0,token1), 0) + pretoken_freq

        # sorted_counts= dict(sorted(sorted_counts.items(),key=lambda x:(x[1],(vocab[x[0][0]],vocab[x[0][1]])),reverse=True))
        # logger.debug(f"Sorted dict is {sorted_counts}")
        # candidate = next(iter(sorted_counts.items()))
        candidate = (best_token_pair,best_count)
        # logger.debug(f"Candidate {candidate}")
        return candidate
        # return (max_key,max_val)

    def apply_top_merge(token_pair,freq):
        new_token_id = len(vocab)
        # the byte string concatenation
        vocab[new_token_id] = vocab[token_pair[0]] + vocab[token_pair[1]]

        # make sure sorted_counts doesn't contain an obsolete pair\
        # if token_pair in sorted_counts:
            # logger.debug(f"Pruning from sorted counts")
            # del sorted_counts[token_pair]
            # logger.debug(f"Sorted dict is {sorted_counts}")

        # logger.debug(f"pretoken size {len(tokens_by_pretoken)}")



        for pretoken,token_ints in tokens_by_pretoken.items():

            debug_mode = token_ints == DEBUG_TOKEN_INTS
            pretoken_frequency = counts_by_pretoken[pretoken]
            # logger.debug(f"Pretoken {pretoken} with frequency {pretoken_frequency}")
            # token_int_len = len(token_ints)
            insertion_indices = []
            insertion_indices_len = 0
            # first sweep through and gather the indices that would need insertions of the new token id
            counter = 0
            for (token_int0, token_int1) in itertools.pairwise(token_ints):
                if token_int0 == token_pair[0] and token_int1 == token_pair[1]:
                    insertion_indices.append(counter)
                    insertion_indices_len += 1
                counter += 1
            token_int_len = counter + 1
            # for counter in range(token_int_len-1):
            #     if token_ints[counter]==token_pair[0] and token_ints[counter+1]==token_pair[1]:
            #         insertion_indices.append(counter)
            #         insertion_indices_len+=1
            # skip this pre token if nothing is to be replaced

            # logger.debug(f"Insertion indices {insertion_indices}")
            # prev_match = -1
            if insertion_indices_len == 0:
                continue

            if CHECK_INVARIANTS:
                new_list = []
                counter = 0
                merge = False
                # logger.debug(f"token ints {token_ints} token_pair {token_pair}")

                if token_pair[0] in token_ints or token_pair[1] in token_ints:
                    while counter < token_int_len:
                        if counter<(token_int_len-1) and token_ints[counter] == token_pair[0] and token_ints[counter+1] == token_pair[1]:
                            new_list.append(new_token_id)
                            counter+=2
                            merge = True
                        else:
                            new_list.append(token_ints[counter])
                            counter+=1
                if merge:
                    # logger.debug(f"Pretoken {pretoken} with token ints {token_ints }")
                    # logger.debug(f" New list {new_list }")

                    if debug_mode:
                        logger.debug(f"Old list {token_ints} New list {new_list}")
                        before = get_counts_by_token_pair(tokens_by_pretoken,counts_by_pretoken)
                        tokens_by_pretoken[pretoken] = new_list
                        after = get_counts_by_token_pair(tokens_by_pretoken,counts_by_pretoken)
                        debug_dicts(before,after)
                        #logger.debug(f"[48,48] is {incremental_counts_by_token_pair[(48, 48)]}")
                        #assert(False)
                    else:
                        tokens_by_pretoken[pretoken] = new_list



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
                insertion_indices_pruned_len = 0
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
                            insertion_indices_pruned_len+=1
                            anchor_idx = target_idx
                        target_idx+=1
                logger.debug(f"Insertion indices_len {insertion_indices_len} Counter {counter} Old insertion indices {insertion_indices} and new one {insertion_indices_pruned}")
                insertion_indices = insertion_indices_pruned
                insertion_indices_len = insertion_indices_pruned_len

            # break


            # decrementing pairs straddling only of the elements of token_pair
            # don't double count for decrements
            # this set is ((token0,token1),index)
            deltas_set = set()
            for insertion_index in insertion_indices:
                # handle case where at very left
                if insertion_index == 0:
                    if token_int_len > 2:
                        t0, t1 =  token_ints[insertion_index + 1], token_ints[insertion_index + 2]
                        if ((t0,t1),(insertion_index + 1)) not in deltas_set:
                            # logger.debug(f"At very left, decrementing right neighbor {(t0, t1)} at by {pretoken_frequency}")
                            incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) - pretoken_frequency
                            if incremental_counts_by_token_pair[(t0,t1)] == 0:
                                # logger.debug(f"HOORAY {(t0,t1)} WENT TO ZERO")
                                del incremental_counts_by_token_pair[(t0,t1)]
                        else:
                            logger.debug(f"edge case {insertion_index + 1} with right neighbor was already decremented")
                        deltas_set.add(((t0,t1),insertion_index+1))
                elif insertion_index == (token_int_len - 2):
                    # at very right
                    if token_int_len > 2:
                        t0, t1 = token_ints[insertion_index -1], token_ints[insertion_index ]
                        if ((t0,t1),(insertion_index -1)) not in deltas_set:
                            # logger.debug(f"At very right, decrementing left neighbor {(t0, t1)} at by {pretoken_frequency}")
                            incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) - pretoken_frequency
                            if incremental_counts_by_token_pair[(t0,t1)] == 0:
                                # logger.debug(f"HOORAY {(t0,t1)} WENT TO ZERO")
                                del incremental_counts_by_token_pair[(t0,t1)]
                        else:
                            logger.debug(f"edge case {(insertion_index -1)} with left neighbor was already decremented")
                        deltas_set.add(((t0,t1),insertion_index -1))
                # in the moddle
                else:
                    if token_int_len > 3:
                        # something is flanking on left
                        t0, t1 = token_ints[insertion_index - 1], token_ints[insertion_index]
                        if ((t0,t1),(insertion_index-1)) not in deltas_set:
                            # logger.debug(f"decrementing left neighbor {(t0, t1)} at index {insertion_index - 1} by {pretoken_frequency}")
                            incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) - pretoken_frequency
                            if incremental_counts_by_token_pair[(t0,t1)] == 0:
                                # logger.debug(f"HOORAY {(t0,t1)} WENT TO ZERO")
                                del incremental_counts_by_token_pair[(t0,t1)]
                        else:
                            logger.debug(f"edge case {(insertion_index-1)} with flanking neighbors was already decremented")
                        deltas_set.add(((t0,t1),insertion_index-1))
                        # something is flanking on right
                        t0, t1 = token_ints[insertion_index +1], token_ints[insertion_index+2]
                        if ((t0,t1),(insertion_index +1)) not in deltas_set:
                            # logger.debug(f"decrementing right neighbor {(t0, t1)} at index {insertion_index + 1} by {pretoken_frequency}")
                            incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) - pretoken_frequency
                            if incremental_counts_by_token_pair[(t0,t1)] == 0:
                                # logger.debug(f"HOORAY {(t0,t1)} WENT TO ZERO")
                                del incremental_counts_by_token_pair[(t0,t1)]
                        else:
                            logger.debug(f"edge case {(insertion_index +1)} with flanking neighbors was already decremented")
                        deltas_set.add(((t0,t1),insertion_index +1))
                logger.debug(f"Decrement set {deltas_set}")

            new_list2 = token_ints.copy()
            new_list2_len = token_int_len
            for insertion_index in reversed(insertion_indices):
                del new_list2[insertion_index+1]
                del new_list2[insertion_index]
                new_list2.insert(insertion_index,new_token_id)
                new_list2_len-=1
            if CHECK_INVARIANTS:
                if new_list2 is not None:
                    logger.debug(f"Token pair {token_pair} Old {token_ints} New lists {new_list} {new_list2}")
                    assert(new_list==new_list2)
            tokens_by_pretoken[pretoken] = new_list2
            # incrementing pairs straddling only of the elements of token_pair
            token_int_len = new_list2_len
            deltas_set.clear()
            for counter in range(token_int_len):
                if new_list2[counter] == new_token_id:
                    # handle case where at very left
                    if counter == 0:
                        if token_int_len > 1:
                            t0, t1 =  new_list2[counter], new_list2[counter + 1]
                            if ((t0, t1), (counter)) not in deltas_set:
                                # logger.debug(f"At very left, incrementing right neighbor {(t0, t1)} at by {pretoken_frequency}")
                                incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) + pretoken_frequency
                            deltas_set.add(((t0, t1), counter ))
                    elif counter == (token_int_len - 1):
                        # at very right
                        if token_int_len > 1:
                            t0, t1 = new_list2[counter-1], new_list2[counter]
                            if ((t0, t1), (counter - 1)) not in deltas_set:
                                # logger.debug(f"At very right, incrementing left neighbor {(t0, t1)} at by {pretoken_frequency}")
                                incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) + pretoken_frequency
                            deltas_set.add(((t0, t1), counter-1))
                    # in the moddle
                    else:
                        if token_int_len > 2:
                            # something is flanking on left
                            t0, t1 = new_list2[counter - 1], new_list2[counter]
                            if ((t0, t1), (counter - 1)) not in deltas_set:
                                # logger.debug(f"Incrementing left neighbor {(t0, t1)} at index {counter - 1} by {pretoken_frequency}")
                                incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) + pretoken_frequency
                            deltas_set.add(((t0, t1), counter-1))
                            # something is flanking on right
                            t0, t1 = new_list2[counter ], new_list2[counter+1]
                            if ((t0, t1), (counter )) not in deltas_set:
                                # logger.debug(f"Incrementing right neighbor {(t0, t1)} at index {counter} by {pretoken_frequency}")
                                incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) + pretoken_frequency
                            deltas_set.add(((t0, t1), counter))

            # decrementing all obsoleted pairs
            if insertion_indices_len> 0 and (token_pair[0], token_pair[1]) in incremental_counts_by_token_pair :
                # logger.debug(f"Decrementing pair {token_pair} from {incremental_counts_by_token_pair[(token_pair[0], token_pair[1])]} by {len(insertion_indices) * pretoken_frequency}")
                incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] -= insertion_indices_len * pretoken_frequency
                assert (incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] >= 0)
                if incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] == 0:
                    # logger.debug(f"HOORAY {(token_pair[0], token_pair[1])} WENT TO ZERO")
                    del incremental_counts_by_token_pair[(token_pair[0], token_pair[1])]
            # elif incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] <0:
            #     logger.debug(f"OH NO WENT BELOWZERO")

            if False:
                logger.debug(f"Doing sanity check for pretoken {pretoken}")
                # rebuilding oracle from scratch
                counts_by_token_pair_oracle = get_counts_by_token_pair(tokens_by_pretoken, counts_by_pretoken)
                # placeholder: now I need to take the maintained table and compare here?
                #
                if incremental_counts_by_token_pair != counts_by_token_pair_oracle:
                    debug_dicts(incremental_counts_by_token_pair, counts_by_token_pair_oracle)
                assert (incremental_counts_by_token_pair == counts_by_token_pair_oracle)

        return new_token_id

    def debug_dicts(incremental, truth):
        # first loop through incremental
        for k in incremental:
            if k not in truth:
                logger.debug(f"Ground truth doesn't contain key {k} which should have values {incremental[k]}")
            elif incremental[k] != truth[k]:
                logger.debug(f"For key {k} incremental value is {incremental[k]} and truth is {truth[k]}")
        for k in truth:
            if k not in incremental:
                logger.debug(f"Incremental doesn't contain key {k} which should have values {truth[k]}")


    def apply_merges():
        # sorted_counts= {}
        last_merge = None
        i = 0
        while True:
            logger.debug(f"At iteration {i} last_merge {last_merge}:")
            token_pair,freq= get_top_token_pair(last_merge)
            if freq>0:
                # logger.debug(f" Max pair is {token_pair} with freq {freq} first byte {vocab[token_pair[0]]} and second byte {vocab[token_pair[1]]}")
                merges.append((vocab[token_pair[0]],vocab[token_pair[1]]))
                last_merge = apply_top_merge(token_pair,freq)
                if CHECK_INVARIANTS:
                    # rebuilding oracle from scratch
                    counts_by_token_pair_oracle = get_counts_by_token_pair(tokens_by_pretoken,counts_by_pretoken)
                    # placeholder: now I need to take the maintained table and compare here?
                    #
                    if incremental_counts_by_token_pair!=counts_by_token_pair_oracle:
                        debug_dicts(incremental_counts_by_token_pair,counts_by_token_pair_oracle)
                    assert (incremental_counts_by_token_pair == counts_by_token_pair_oracle)
                # logger.debug(f"Merges {merges}")
                # logger.debug(f"Vocab {vocab}")
            else:
                logger.debug(f"Nothing was merged. Aborting")
                break
            if len(vocab)>=MAX_TOKENS :
                logger.debug(f"Ending with vocab length {len(vocab)} . Aborting")
                break
            i+=1



    # load_file('small.txt')
    #load_file("../tests/fixtures/tinystories_sample_5M.txt")
    load_file(input_path)
    incremental_counts_by_token_pair = get_counts_by_token_pair(tokens_by_pretoken,counts_by_pretoken)
    logger.debug(f"Baseline: {incremental_counts_by_token_pair}")
    apply_merges()
    # raise NotImplementedError
    return (vocab,merges)
    # raise NotImplementedError
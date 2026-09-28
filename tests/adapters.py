from __future__ import annotations

import os
from collections.abc import Iterable
from email import iterators
from itertools import pairwise
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

ord('A')
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


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
                Merges are ordered by order of creation.
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
    print(f"Special tokens are: {special_tokens}")

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

    i=0
    for j in range(256):    #print(f"i is {i} with char {chr(i)}")
        vocab[i]=i.to_bytes()
        i+=1
    for special_token in special_tokens:
        vocab[i]=special_token.encode("utf-8")
        i+=1

    print(f"Initial vocab dict {vocab}")

    # a list of token ID pairs, sorted by their counts in descending order

    # sorted_counts = {}
    # print(f"Sorted counts initialized length {len(sorted_counts)}")


    def load_file(filename):
        with open(filename, 'r') as f:
            print(f"Loading file {filename}")
            file_content = f.read()
            regex_pattern = "|".join(map(re.escape,special_tokens))
            # only taking the first 'document' for prototyping. when things are working, will generalize remaining code into a function and loop over elements
            file_content_docs = re.split(regex_pattern,file_content)
            for file_content in file_content_docs:
                #print(f"regex pattern is {regex_pattern} Filtered {file_content}")
                pretoken_iter = re.finditer(PAT, file_content)
                for pretoken in pretoken_iter:
                    pretoken_str = pretoken.group(0)
                    counts_by_pretoken[pretoken_str] = counts_by_pretoken.get(pretoken_str, 0) + 1


        #vocab_set.add('<|endoftext|>')
        #vocab_list = list(vocab_set)
        #vocab_list = [for item in vocab_set]
        # print(f"Vocab dict {vocab}")
        # print(f"counts_by_pretoken {counts_by_pretoken}")
        for pretoken in counts_by_pretoken.keys():
            tokens_by_pretoken[pretoken] = list(pretoken.encode("utf-8"))
        # print(f"tokens_by_pretoken {tokens_by_pretoken}")

    def get_counts_by_token_pair(tokens_by_pretoken, counts_by_pretoken):
        counts_by_tokenpair = {}
        for pretoken in tokens_by_pretoken.keys():
            pretoken_freq = counts_by_pretoken[pretoken]
            # print(f" pretoken: {pretoken} freq:{pretoken_freq}")
            token_list = tokens_by_pretoken[pretoken]#token_pair_frequency = {}
            # print(f"last merge {last_merge} token_list {token_list}")
            # if last_merge is None:
                # print(f"Exhaustive count of all pairs")
                # last_pair = None
            for (token0,token1) in itertools.pairwise(token_list):
            # for (token0,token1) in zip(token_list,token_list[1:]):
                # if COUNT_OVERLAPS:
                #sorted_counts[(token0,token1)] = sorted_counts.get((token0,token1),0) + pretoken_freq
                # print(f"Token {token0} {token1}")
                counts_by_tokenpair[(token0,token1)] = counts_by_tokenpair.get((token0,token1),0) + pretoken_freq
                    # else:
                    #     if (last_pair is not None and last_pair!=(token0,token1)):

                    # last_pair = (token0,token1)
        return counts_by_tokenpair


    def get_top_token_pair(last_merge):

        # this data structure should contain the key as the tuple of token0 and token 1 and the value as a tuple of the counts, and a list of pre tokens it belong to
        counts_by_tokenpair = None
        if CHECK_INVARIANTS:
            print("Launching slow version to compare fast and slow versions")
            counts_by_tokenpair = get_counts_by_token_pair(tokens_by_pretoken,counts_by_pretoken)
        else:
            print("Launching express version to compare fast and slow versions")
            counts_by_tokenpair = incremental_counts_by_token_pair


        best_count = -1
        best_token_pair = None
        best_byte_pair = None
        for tokenpair,freq in counts_by_tokenpair.items():
            if freq > best_count:
                best_count = freq
                best_token_pair = tokenpair
                # print(f"fetching vocab index {tokenpair[0]} and {tokenpair[1]}")
                best_byte_pair = (vocab[tokenpair[0]],vocab[tokenpair[1]])
            elif freq == best_count and  (vocab[tokenpair[0]],vocab[tokenpair[1]]) > best_byte_pair:
                best_count = freq
                best_token_pair = tokenpair
                best_byte_pair = (vocab[tokenpair[0]],vocab[tokenpair[1]])

            # else:
            #     if last_merge in token_list:
            #         print(f"Incremental logic on {token_list}")
            #         for (token0,token1) in zip(token_list,token_list[1:]):
            #             #token_pair_frequency[(token0,token1)] = token_pair_frequency.get((token0,token1), 0) + pretoken_freq
            #             if token0==last_merge or token1==last_merge:
            #                 # print(f"Incremental logic between {token0} and {token1}")
            #                 # if not COUNT_OVERLAPS and last_pair is not None and last_pair==(token0,token1):
            #                 #         print(f"token pair {token0,token1}")
            #
            #
            #                 sorted_counts[(token0,token1)] = sorted_counts.get((token0,token1),0) + pretoken_freq

                        #sorted_counts[(token0,token1)]  = token_pair_frequency.get((token0,token1), 0) + pretoken_freq

        # sorted_counts= dict(sorted(sorted_counts.items(),key=lambda x:(x[1],(vocab[x[0][0]],vocab[x[0][1]])),reverse=True))
        # print(f"Sorted dict is {sorted_counts}")
        # candidate = next(iter(sorted_counts.items()))
        candidate = (best_token_pair,best_count)
        # print(f"Candidate {candidate}")
        return candidate
        # return (max_key,max_val)

    def apply_top_merge(token_pair,freq):
        new_token_id = len(vocab)
        # the byte string concatenation
        vocab[new_token_id] = vocab[token_pair[0]] + vocab[token_pair[1]]

        # make sure sorted_counts doesn't contain an obsolete pair\
        # if token_pair in sorted_counts:
            # print(f"Pruning from sorted counts")
            # del sorted_counts[token_pair]
            # print(f"Sorted dict is {sorted_counts}")

        # print(f"pretoken size {len(tokens_by_pretoken)}")




        for pretoken,token_ints in tokens_by_pretoken.items():
            debug_mode = token_ints == DEBUG_TOKEN_INTS
            pretoken_frequency = counts_by_pretoken[pretoken]
            # print(f"Pretoken {pretoken} with frequency {pretoken_frequency}")
            token_int_len = len(token_ints)

            if CHECK_INVARIANTS:
                new_list = []
                counter = 0
                merge = False
                # print(f"token ints {token_ints} token_pair {token_pair}")

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
                    # print(f"Pretoken {pretoken} with token ints {token_ints }")
                    # print(f" New list {new_list }")

                    if debug_mode:
                        print(f"Old list {token_ints} New list {new_list}")
                        before = get_counts_by_token_pair(tokens_by_pretoken,counts_by_pretoken)
                        tokens_by_pretoken[pretoken] = new_list
                        after = get_counts_by_token_pair(tokens_by_pretoken,counts_by_pretoken)
                        debug_dicts(before,after)
                        #print(f"[48,48] is {incremental_counts_by_token_pair[(48, 48)]}")
                        #assert(False)
                    else:
                        tokens_by_pretoken[pretoken] = new_list


            # new logic
            # look for boundary cases

            # first case is where token0 and token1 are the same which can cause overlapping merges
            # if token_pair[0]==token_pair[1]:
            #     prev_matches = 0
            #     for counter in range(token_int_len):
            #         if counter>0:
            #             if token_ints[counter-1]==token_ints[counter]:
            #                 prev_matches+=1
            #             else:
            #                 prev_matches = 0
            #         if prev_matches > 1:
            #             print(f"edge case of overlapping: {token_ints}")
            #             assert (False)
            #
            insertion_indices = []
            # first sweep through and gather the indices that would need insertions of the new token id
            for counter in range(token_int_len-1):
                if token_ints[counter]==token_pair[0] and token_ints[counter+1]==token_pair[1]:
                    insertion_indices.append(counter)
            # skip this pre token if nothing is to be replaced
            if len(insertion_indices)==0:
                continue
            # print(f"Insertion indices {insertion_indices}")
            # prev_match = -1
            insertion_indices_len =len(insertion_indices)
            delta = 3
            for counter in range(insertion_indices_len):
                if counter > 0:
                    delta = insertion_indices[counter] - insertion_indices[counter-1]
                    if delta == 1:
                        print(f"edge case of overlapping: {token_ints}")
                        break
                        # assert (False)
                    elif delta == 2:
                        print(f"edge case of neighboring pairs: {token_ints}")
                        break
                        # assert(False)

            if delta ==1:
                insertion_indices_pruned = []
                insertion_indices_pruned.append(insertion_indices[0])
                anchor_value = insertion_indices[0]
                anchor_idx = 0
                for walker in range( insertion_indices_len-1):
                    print(f"Looking from {anchor_idx+1} to {insertion_indices_len}")
                    target_idx = walker+1
                    while target_idx< insertion_indices_len:
                        print(f"Comparing {insertion_indices[anchor_idx]} at position {anchor_idx} to {insertion_indices[target_idx]} at position {(target_idx)}")
                        if (insertion_indices[target_idx] - insertion_indices[anchor_idx]) > 1:
                            insertion_indices_pruned.append(insertion_indices[target_idx])
                            anchor_idx = target_idx
                        target_idx+=1
                print(f"Insertion indices_len {insertion_indices_len} Counter {counter} Old insertion indices {insertion_indices} and new one {insertion_indices_pruned}")
                insertion_indices = insertion_indices_pruned
                insertion_indices_len = len(insertion_indices)
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
                            # print(f"At very left, decrementing right neighbor {(t0, t1)} at by {pretoken_frequency}")
                            incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) - pretoken_frequency
                            if incremental_counts_by_token_pair[(t0,t1)] == 0:
                                # print(f"HOORAY {(t0,t1)} WENT TO ZERO")
                                del incremental_counts_by_token_pair[(t0,t1)]
                        else:
                            print(f"edge case {insertion_index + 1} with right neighbor was already decremented")
                        deltas_set.add(((t0,t1),insertion_index+1))
                elif insertion_index == (token_int_len - 2):
                    # at very right
                    if token_int_len > 2:
                        t0, t1 = token_ints[insertion_index -1], token_ints[insertion_index ]
                        if ((t0,t1),(insertion_index -1)) not in deltas_set:
                            # print(f"At very right, decrementing left neighbor {(t0, t1)} at by {pretoken_frequency}")
                            incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) - pretoken_frequency
                            if incremental_counts_by_token_pair[(t0,t1)] == 0:
                                # print(f"HOORAY {(t0,t1)} WENT TO ZERO")
                                del incremental_counts_by_token_pair[(t0,t1)]
                        else:
                            print(f"edge case {(insertion_index -1)} with left neighbor was already decremented")
                        deltas_set.add(((t0,t1),insertion_index -1))
                # in the moddle
                else:
                    if token_int_len > 3:
                        # something is flanking on left
                        t0, t1 = token_ints[insertion_index - 1], token_ints[insertion_index]
                        if ((t0,t1),(insertion_index-1)) not in deltas_set:
                            # print(f"decrementing left neighbor {(t0, t1)} at index {insertion_index - 1} by {pretoken_frequency}")
                            incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) - pretoken_frequency
                            if incremental_counts_by_token_pair[(t0,t1)] == 0:
                                # print(f"HOORAY {(t0,t1)} WENT TO ZERO")
                                del incremental_counts_by_token_pair[(t0,t1)]
                        else:
                            print(f"edge case {(insertion_index-1)} with flanking neighbors was already decremented")
                        deltas_set.add(((t0,t1),insertion_index-1))
                        # something is flanking on right
                        t0, t1 = token_ints[insertion_index +1], token_ints[insertion_index+2]
                        if ((t0,t1),(insertion_index +1)) not in deltas_set:
                            # print(f"decrementing right neighbor {(t0, t1)} at index {insertion_index + 1} by {pretoken_frequency}")
                            incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) - pretoken_frequency
                            if incremental_counts_by_token_pair[(t0,t1)] == 0:
                                # print(f"HOORAY {(t0,t1)} WENT TO ZERO")
                                del incremental_counts_by_token_pair[(t0,t1)]
                        else:
                            print(f"edge case {(insertion_index +1)} with flanking neighbors was already decremented")
                        deltas_set.add(((t0,t1),insertion_index +1))
                print(f"Decrement set {deltas_set}")

            new_list2 = token_ints.copy()
            for insertion_index in reversed(insertion_indices):
                del new_list2[insertion_index+1]
                del new_list2[insertion_index]
                new_list2.insert(insertion_index,new_token_id)
            if CHECK_INVARIANTS:
                if new_list2 is not None:
                    print(f"Token pair {token_pair} Old {token_ints} New lists {new_list} {new_list2}")
                    assert(new_list==new_list2)
            tokens_by_pretoken[pretoken] = new_list2
            # incrementing pairs straddling only of the elements of token_pair
            token_int_len = len(new_list2)
            deltas_set.clear()
            for counter in range(token_int_len):
                if new_list2[counter] == new_token_id:
                    # handle case where at very left
                    if counter == 0:
                        if token_int_len > 1:
                            t0, t1 =  new_list2[counter], new_list2[counter + 1]
                            if ((t0, t1), (counter)) not in deltas_set:
                                # print(f"At very left, incrementing right neighbor {(t0, t1)} at by {pretoken_frequency}")
                                incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) + pretoken_frequency
                            deltas_set.add(((t0, t1), counter ))
                    elif counter == (token_int_len - 1):
                        # at very right
                        if token_int_len > 1:
                            t0, t1 = new_list2[counter-1], new_list2[counter]
                            if ((t0, t1), (counter - 1)) not in deltas_set:
                                # print(f"At very right, incrementing left neighbor {(t0, t1)} at by {pretoken_frequency}")
                                incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) + pretoken_frequency
                            deltas_set.add(((t0, t1), counter-1))
                    # in the moddle
                    else:
                        if token_int_len > 2:
                            # something is flanking on left
                            t0, t1 = new_list2[counter - 1], new_list2[counter]
                            if ((t0, t1), (counter - 1)) not in deltas_set:
                                # print(f"Incrementing left neighbor {(t0, t1)} at index {counter - 1} by {pretoken_frequency}")
                                incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) + pretoken_frequency
                            deltas_set.add(((t0, t1), counter-1))
                            # something is flanking on right
                            t0, t1 = new_list2[counter ], new_list2[counter+1]
                            if ((t0, t1), (counter )) not in deltas_set:
                                # print(f"Incrementing right neighbor {(t0, t1)} at index {counter} by {pretoken_frequency}")
                                incremental_counts_by_token_pair[(t0, t1)] = incremental_counts_by_token_pair.get((t0, t1),0) + pretoken_frequency
                            deltas_set.add(((t0, t1), counter))

            # decrementing all obsoleted pairs
            if len(insertion_indices) > 0 and (token_pair[0], token_pair[1]) in incremental_counts_by_token_pair :
                # print(f"Decrementing pair {token_pair} from {incremental_counts_by_token_pair[(token_pair[0], token_pair[1])]} by {len(insertion_indices) * pretoken_frequency}")
                incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] -= len(
                    insertion_indices) * pretoken_frequency
                assert (incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] >= 0)
                if incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] == 0:
                    # print(f"HOORAY {(token_pair[0], token_pair[1])} WENT TO ZERO")
                    del incremental_counts_by_token_pair[(token_pair[0], token_pair[1])]
            # elif incremental_counts_by_token_pair[(token_pair[0], token_pair[1])] <0:
            #     print(f"OH NO WENT BELOWZERO")

            if False:
                print(f"Doing sanity check for pretoken {pretoken}")
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
                print(f"Ground truth doesn't contain key {k} which should have values {incremental[k]}")
            elif incremental[k] != truth[k]:
                print(f"For key {k} incremental value is {incremental[k]} and truth is {truth[k]}")
        for k in truth:
            if k not in incremental:
                print(f"Incremental doesn't contain key {k} which should have values {truth[k]}")


    def apply_merges():
        # sorted_counts= {}
        last_merge = None
        i = 0
        while True:
            print(f"At iteration {i} last_merge {last_merge}:")
            token_pair,freq= get_top_token_pair(last_merge)
            if freq>0:
                # print(f" Max pair is {token_pair} with freq {freq} first byte {vocab[token_pair[0]]} and second byte {vocab[token_pair[1]]}")
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
                # print(f"Merges {merges}")
                # print(f"Vocab {vocab}")
            else:
                print(f"Nothing was merged. Aborting")
                break
            if len(vocab)>=MAX_TOKENS :
                print(f"Ending with vocab length {len(vocab)} . Aborting")
                break
            i+=1



    # load_file('small.txt')
    #load_file("../tests/fixtures/tinystories_sample_5M.txt")
    load_file(input_path)
    incremental_counts_by_token_pair = get_counts_by_token_pair(tokens_by_pretoken,counts_by_pretoken)
    print(f"Baseline: {incremental_counts_by_token_pair}")
    apply_merges()
    # raise NotImplementedError
    return (vocab,merges)
    # raise NotImplementedError

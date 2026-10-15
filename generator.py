import sys
import argparse
import yaml
from pathlib import Path
import logging

import numpy as np
import torch

from cs336_basics import io_utils,custom_modules,custom_training,tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create a console handler
console_handler = logging.StreamHandler()
# Create a formatting layout
formatter = logging.Formatter('%(name)s - %(levelname)s - %(asctime)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to your logger
logger.addHandler(console_handler)

def main():
    device = "cpu"
    parser = argparse.ArgumentParser(description="A language model generator")
    parser.add_argument("--model_params", required=True, help="YAML file for model parameters")
    parser.add_argument("--merges_filename",required=True, help="merges text file from byte pair trainer")
    parser.add_argument("--max_tokens_generated", required=True, help="maximum tokens to generate", default=100)
    parser.add_argument("--temperature", required=True, help="temperature, close to zero is deterministic above 1 is more stochastic", default=1)
    parser.add_argument("--softmax_nucleus_threshold", required=True, help="probability threshold for softmax. below this value probs are zeroed out")
    parser.add_argument("--end_of_text", required=True,help="end of text string", default = "|<endoftext>|")

    args = parser.parse_args()
    logger.info(f"Command line arguments are {args}")


    model_params = None
    with open(args.model_params,"r") as f_in:
        yaml_data = yaml.safe_load(f_in)
        model_params = yaml_data['model_params']

    if model_params is None:
        logger.error(f"Could not load model config file {args.model_params}")
        sys.exit(1)



    vocab_file = model_params['vocab_file']
    checkpoint_file = model_params['checkpoint_file']
    context_length = model_params['context_length']
    embedding_dim = model_params['embedding_dim']
    num_layers = model_params['num_layers']
    num_heads = model_params['num_heads']
    feed_forward_dim = model_params['feed_forward_dim']
    rope_theta = model_params['rope_theta']

    special_tokens = [args.end_of_text]

    # get the vocab size
    vocab = io_utils.deserialize_vocab(vocab_file)
    vocab_size = len(vocab)

    # get the merges
    merges = io_utils.deserialize_merges(args.merges_filename)

    #tokenizer
    tkn = tokenizer.Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    # create the language model
    model:torch.nn.Module = custom_modules.TransformerLm(vocab_size=vocab_size,
                                                  context_length=context_length,
                                                  num_layers=num_layers,
                                                  d_model=embedding_dim,
                                                  num_heads=num_heads,
                                                  d_ff=feed_forward_dim,
                                                  rope_theta=rope_theta
                                                  )

    model = torch.compile(model)

    cross_entropy_loss = custom_modules.CrossEntropyLoss()

    # create the optimizer
    optimizer = custom_training.AdamW(params=model.parameters(),lr=1e-3,
                                weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    softmax = custom_modules.Softmax(temperature=float(args.temperature))



    t:int = 0

    checkpoint_path = Path(checkpoint_file)
    if checkpoint_path.is_file():
        t = custom_training.run_load_checkpoint(src=checkpoint_path,model=model,optimizer=optimizer)
        logger.info(f"Loading checkpoint file {checkpoint_path} and starting at iteration {t}")



    def get_end_of_text_id(vocab: dict[int, bytes], end_of_text: str) -> int:
        eot_bytes = end_of_text.encode('utf-8')
        for k, v in vocab.items():
            if v == eot_bytes:
                return k
        return -1

    end_of_text_id = get_end_of_text_id(vocab=vocab, end_of_text=args.end_of_text)
    assert (end_of_text_id > -1)

    def mask_weights(weights: torch.Tensor, softmax_nucleus_threshold: float = 1.0):
        values, indices = torch.sort(weights, descending=True)
        # logger.debug(f"sorted {values.shape} indices {indices.shape}")
        last_index = weights.shape[-1]
        i = 0
        sum = 0
        # logger.debug(f"Weight shape is {weights.shape}")
        mask = torch.zeros(weights.shape, dtype=bool, requires_grad=False)
        while i < last_index:
            # logger.debug(f"values shape {values.shape}")
            sum += values[i]
            # logger.debug(f"sum is {sum} at {i}")
            mask[indices[i]] = True
            if sum >= softmax_nucleus_threshold:
                break
            i += 1
        # print(f"mask {mask}")
        weight_masked = torch.where(mask, weights, 0)
        return weight_masked

    while(True):
        prompt = input("Enter your prompt:")
        token_ids = tkn.encode(prompt)
        seq_len = min(context_length,len(token_ids))
        token_ids = token_ids[-seq_len:]
        prompt_len = len(token_ids)
        model_input = torch.tensor(token_ids, dtype=torch.long).view(1, prompt_len).to(device)
        tokens_generated = model_input[0].tolist()
        output = tkn.decode(tokens_generated)
        print(output,end="")

        token_iter = 0
        best_token = -1

        while best_token != end_of_text_id and token_iter < int(args.max_tokens_generated):
            # logger.debug(f"At iteration {token_iter}")

            # logger.debug(f"model_input {model_input}")
            model_output = model(model_input)
            probs = softmax(in_features=model_output, dim=1)
            last_pos_prob = probs[0, -1]
            probs = mask_weights(weights=last_pos_prob, softmax_nucleus_threshold=float(args.softmax_nucleus_threshold))


            last_index_sampled = torch.multinomial(probs, num_samples=1, replacement=True)
            model_input = torch.cat((model_input[..., 1:], last_index_sampled.view(1, -1)), dim=-1)
            # logger.debug(f"last_index_sampled {last_index_sampled} Input is now {model_input}")
            best_token = last_index_sampled.item()
            print(tkn.decode([best_token]),end="",flush=True)
            token_iter += 1

        output = tkn.decode(tokens_generated)
        print()

if __name__=="__main__":
    main()

import time
import numpy as np
from sympy import to_number_field
import sys
import yaml
import logging
import argparse

from cs336_basics.tokenizer import Tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create a console handler
console_handler = logging.StreamHandler()
# Create a formatting layout
formatter = logging.Formatter('%(name)s - %(levelname)s - %(asctime)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to your logger
logger.addHandler(console_handler)

def main():
    parser = argparse.ArgumentParser(description="A byte pair encoding trainer")
    parser.add_argument("--model_params", required=True, help="YAML file for model parameters")
    args = parser.parse_args()
    logger.info(f"Command line arguments are {args}")

    model_params = None
    with open(args.model_params, "r") as f_in:
        yaml_data = yaml.safe_load(f_in)
        model_params = yaml_data['model_params']

    if model_params is None:
        logger.error(f"Could not load model config file {args.model_params}")
        sys.exit(1)

    special_tokens = model_params['special_tokens']
    merges_file = model_params['merges_file']
    vocab_file = model_params['vocab_file']
    train_in_file = model_params['train_in_file']
    train_out_file = model_params['train_out_file']
    valid_in_file = model_params['valid_in_file']
    valid_out_file = model_params['valid_out_file']

    tokenizer = Tokenizer.from_files(vocab_file, merges_file, [special_tokens])

    def serialize_token_ids(infile:str,outfile:str,tokenizer:Tokenizer):
        logger.info(f"Processing {infile} to {outfile}")
        with open(infile,'r') as f:
            stream = (_id for _id in tokenizer.encode_iterable(f))
            # byte_buffer = array.array('H',owt_train_stream).tobytes()
            uint16_array = np.fromiter(stream,dtype=np.uint16)
            np.save(outfile,uint16_array)
            print(f"Numpy array created of size {len(uint16_array)}")

    serialize_token_ids(train_in_file, train_out_file, tokenizer)
    serialize_token_ids(valid_in_file, valid_out_file, tokenizer)

if __name__=="__main__":
    main()

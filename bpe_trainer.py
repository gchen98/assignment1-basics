import sys
import yaml
import logging
import argparse


from cs336_basics import bpe,io_utils

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

    workers = int(model_params['workers'])
    input_path = model_params['input_path']
    vocab_size = int(model_params['vocab_size'])
    outmerges = model_params['outmerges']
    outvocab = model_params['outvocab']
    special_tokens = model_params['special_tokens']

    # workers=4
    # input_path = "data/TinyStoriesV2-GPT4-train.txt"
    # input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    # vocab_size = 10000
    # outmerges = "tinystories_merges.txt"
    # outvocab = "tinystories_vocab.json"

    # special_tokens=['<|endoftext|>']

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

    # workers=24
    # input_path = "data/owt_train.txt"
    # vocab_size = 32000
    # outmerges = "owt_merges.txt"
    # outvocab = "owt_vocab.json"
    logger.info(f"input path {input_path} vocab size {vocab_size} workers {workers}")
    (vocab,merges) = bpe.run_train_bpe(input_path,vocab_size,[special_tokens],workers)
    logger.info(f"vocab is {vocab} and merges is {merges}")
    io_utils.serialize_merges(outmerges,merges)
    io_utils.serialize_vocab(outvocab,vocab)


if __name__=="__main__":
    main()

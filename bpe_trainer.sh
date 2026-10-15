config_file='data/bpe_tinystories_config.yaml'
#config_file='data/bpe_corpus_en_config.yaml'
uv run python bpe_trainer.py --model_params ${config_file}

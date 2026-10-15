import sys
import argparse
import time

import yaml
from pathlib import Path
import logging

import numpy as np
import torch

from cs336_basics import io_utils
from cs336_basics import custom_modules
from cs336_basics import custom_training

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
    parser = argparse.ArgumentParser(description="A language model trainer")
    parser.add_argument("--model_params", required=True, help="YAML file for model parameters")
    parser.add_argument("--batch_size", required=True, help="number of training examples to sample randomly")
    parser.add_argument("--dataset",required=True, help="serialized numpy file of token ID dataset")
    parser.add_argument("--validation", required=True, help="serialized numpy file of validation token ID dataset")
    parser.add_argument("--metrics_file", required=True, help="various statistics will be saved here.")
    parser.add_argument("--steps", required=True, help="optimization steps", default=1000)
    parser.add_argument("--checkpoint_interval", required=True, help="number of steps before writing checkpoint file",default=100)
    parser.add_argument("--max_lr", required=True, help="maximum learning rate")
    parser.add_argument("--min_lr", required=True, help="minimum learning rate")
    parser.add_argument("--warmup_steps", required=True, help="warm up steps to take before cosine decay learning rate")
    parser.add_argument("--max_gradient_norm", required=True, help="L2 norm of gradients threshold for clipping.")

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
    unopt_checkpoint_file = model_params['unopt_checkpoint_file']
    opt_checkpoint_file = model_params['opt_checkpoint_file']
    context_length = model_params['context_length']
    embedding_dim = model_params['embedding_dim']
    num_layers = model_params['num_layers']
    num_heads = model_params['num_heads']
    feed_forward_dim = model_params['feed_forward_dim']
    rope_theta = model_params['rope_theta']


    steps = int(args.steps)
    check_point_interval = int(args.checkpoint_interval)

    # get the vocab size
    vocab = io_utils.deserialize_vocab(vocab_file)
    vocab_size = len(vocab)

    # create the language model
    orig_model:torch.nn.Module = custom_modules.TransformerLm(vocab_size=vocab_size,
                                                  context_length=context_length,
                                                  num_layers=num_layers,
                                                  d_model=embedding_dim,
                                                  num_heads=num_heads,
                                                  d_ff=feed_forward_dim,
                                                  rope_theta=rope_theta
                                                  )

    model = torch.compile(orig_model)

    cross_entropy_loss = custom_modules.CrossEntropyLoss()

    # create the optimizer
    optimizer = custom_training.AdamW(params=model.parameters(),lr=1e-3,
                                weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)



    # load the dataset
    dataset = np.load(args.dataset, mmap_mode='r')
    validation = np.load(args.validation, mmap_mode = 'r')

    t:int = 0
    batch_size = int(args.batch_size)


    opt_checkpoint_path = Path(opt_checkpoint_file)
    if opt_checkpoint_path.is_file():
        t = custom_training.run_load_checkpoint(src=opt_checkpoint_path,model=model,optimizer=optimizer)
        logger.info(f"Loading checkpoint file {opt_checkpoint_path} and starting at iteration {t}")

    unopt_checkpoint_path = Path(unopt_checkpoint_file)
        # model = torch.compile(model)
        # custom_training.run_save_checkpoint(model=model, optimizer=optimizer, iteration=t, out="data/opt_tinystories_checkpoint.pt")
        # sys.exit(0)

    logger.info(f"Beginning training loop, will checkpoint at interval of {check_point_interval}")
    metrics_out = open(args.metrics_file,"a")
    metrics_out.write("ITERATION\tLEARNING_RATE\tTRAINING_LOSS\tVALIDATION_LOSS\n")
    metrics_out.flush()
    for t in range(t,steps):
        iteration_start = time.perf_counter()
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = custom_training.get_learning_rate_schedule(current_step = t,lr_max = float(args.max_lr),
                                                                lr_min = float(args.min_lr),
                                                                warmup_steps = int(args.warmup_steps),
                                                                post_annealing_step = steps)
            logger.debug(f"Learning rates: old {old_lr} new {new_lr}")
            param_group['lr'] =new_lr

        inputs, targets = custom_training.get_batch(dataset=dataset, batch_size=batch_size,
                                                    context_length=context_length, device="cpu")


        optimizer.zero_grad()  # Reset the gradients for all learnable parameters
        model_output = model(inputs)
        loss =  cross_entropy_loss(logits=model_output, targets=targets)
        # loss = (weights ** 2).mean() # Compute a scalar loss value.
        logger.info(f"Loss is {loss.cpu().item()}")
        loss.backward()  # Run backward pass, which computes gradients

        # ssq: float = 0
        # for parameter in model.parameters():
        #     if parameter.grad is not None:
        #         ssq += torch.sum(parameter.grad ** 2)
        # l2_norm = torch.sqrt(ssq)
        l2_norm = custom_training.clip_gradient(parameters = model.parameters(),max_l2_norm = float(args.max_gradient_norm))
        logger.debug(f"L2 norm of parameters is {l2_norm}")

        optimizer.step()
        iteration_end = time.perf_counter()
        iteration_time = (iteration_end-iteration_start)
        logger.info(f"Training iteration {t} took {iteration_time} seconds")
        if t % check_point_interval == 0:
            logger.info(f"Saving checkpoint at interval {t}")
            custom_training.run_save_checkpoint(model=model, optimizer=optimizer, iteration=t, out=opt_checkpoint_path)
            custom_training.run_save_checkpoint(model=orig_model, optimizer=optimizer, iteration=t, out=unopt_checkpoint_path)
            v_inputs, v_targets = custom_training.get_batch(dataset=validation, batch_size=batch_size,
                                                            context_length=context_length, device="cpu")
            validation_output = model(v_inputs)
            validation_loss = cross_entropy_loss(logits=validation_output, targets=v_targets)
            logger.info(f"Validation Loss is {validation_loss.cpu().item()}")
        metrics_out.write(f"{t}\t{old_lr}\t{loss}\t{validation_loss}\n")
        metrics_out.flush()


    custom_training.run_save_checkpoint(model =model,optimizer=optimizer,iteration = t,out=opt_checkpoint_path)
    custom_training.run_save_checkpoint(model=orig_model, optimizer=optimizer, iteration=t, out=unopt_checkpoint_path)
    metrics_out.close()


if __name__=="__main__":
    main()

import math

import os
from typing import BinaryIO,IO
import numpy as np
import torch
import logging

from collections.abc import Callable, Iterable
from einops import rearrange,einsum,repeat,reduce
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create a console handler
console_handler = logging.StreamHandler()
# Create a formatting layout
formatter = logging.Formatter('%(name)s - %(levelname)s - %(asctime)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to your logger
logger.addHandler(console_handler)

class SGD(torch.optim.Optimizer):
    def __init__(self,params, lr = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr}
        super().__init__(params,defaults)

    def step(self,closure: Optional[Callable] = None):
        logger.debug("Calling step")
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p
                t = state.get("t",0) # Get iteration number from the state, or 0
                grad = p.grad.data # Get the gradient of loss with respect to p
                p.data -= lr / math.sqrt(t+1) * grad # Update weight tensor in-place
                state["t"] = t + 1 # Increment iteration number
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self,params = None,
                 lr = 1e-3,weight_decay:float = 0.01,
                 betas:tuple[float,float] = (0.9,0.999),
                 eps:float = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr}
        super().__init__(params,defaults)
        self.betas = betas
        self.eps = eps
        self.first_moments:dict[str,torch.Tensor] = {}
        self.second_moments:dict[str,torch.Tensor] = {}
        for group in self.param_groups:
            # logger.debug(f"group {group}")
            self.first_moments
            for p in group["params"]:
                # logger.debug(f"parameter {p}")
                params_shape = p.shape
                # logger.debug(f"params shape is: {params_shape}")
                self.first_moments[p] = torch.zeros(params_shape)
                self.second_moments[p] = torch.zeros(params_shape)
        # assert(False)
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        logger.debug("AdamW created")

    def step(self,closure: Optional[Callable] = None):
        # logger.debug("Calling step")
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p
                # t = state.get("t",0) # Get iteration number from the state, or 0
                t = state.get("t", 1)
                grad = p.grad.data # Get the gradient of loss with respect to p
                alpha_t = lr * math.sqrt(1-self.betas[1]**t)/(1-self.betas[0]**t)
                p.data -= lr * self.weight_decay * p.data
                self.first_moments[p] = self.betas[0] * self.first_moments[p] + (1-self.betas[0]) * grad
                self.second_moments[p] = self.betas[1] * self.second_moments[p] + (1-self.betas[1]) * grad ** 2
                # p.data -= lr / math.sqrt(t+1) * grad # Update weight tensor in-place
                p.data -= alpha_t * self.first_moments[p] / (torch.sqrt(self.second_moments[p]) + self.eps )# Update weight tensor in-place
                state["t"] = t + 1 # Increment iteration number
        return loss

def get_learning_rate_schedule(current_step:int,lr_max:float,lr_min:float,
                               warmup_steps:int,post_annealing_step:int)->float:
    if current_step<warmup_steps:
        return (current_step/warmup_steps) * lr_max
    elif current_step<= post_annealing_step:
        return lr_min + 0.5 * (1 + math.cos(((current_step - warmup_steps)/ (post_annealing_step-warmup_steps)) * math.pi)) * (lr_max-lr_min)
    else:
        return lr_min

def clip_gradient(parameters:Iterable[torch.nn.Parameter],max_l2_norm:float):
    eps = 1e-6
    cached_parameters = list(parameters)
    ssq:float = 0
    for parameter in cached_parameters:
        if parameter.grad is not None:
            ssq+=torch.sum(parameter.grad ** 2)
    l2_norm = torch.sqrt(ssq)
    if l2_norm > max_l2_norm:
        logger.debug(f"l2_norm is {l2_norm} max is {max_l2_norm}")
        scaling = max_l2_norm / (l2_norm + eps)
        for parameter in cached_parameters:
            if parameter.grad is not None:
                # logger.debug(f"Before {parameter.grad}")
                parameter.grad *= scaling
                # logger.debug(f"After {parameter.grad}")
def get_batch(dataset:np.array, batch_size:int, context_length: int, device:str)\
        ->tuple[torch.LongTensor,torch.LongTensor]:
    """
    dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.
    :return:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    start_indices = torch.randint(low=0, high=(dataset.size - context_length), size=(batch_size,))
    logger.debug(f"starts: {start_indices}")
    # dataset[start_indices,]
    training = []
    targets = []
    for start_index in start_indices:
        # logger.debug(f"start {start_index}")
        training.append( dataset[start_index:start_index + context_length])
        next_index = start_index + 1
        targets.append(dataset[next_index:next_index + context_length])
    # print(f"Training: \n{training} and targets \n{targets}")

    training_v = torch.tensor( np.concatenate(training,axis=0),dtype=torch.long).view(batch_size, context_length).to(device)
    # training_v = torch.LongTensor(training,device=device).view(batch_size, context_length)
    target_v = torch.tensor(np.concatenate(targets,axis=0), dtype = torch.long).view(batch_size, context_length).to(device)
    # target_v = torch.LongTensor(targets,device=device).view(batch_size, context_length)
    # logger.debug(f"Training v {training_v.shape}")
    return (training_v,target_v)

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
    all_states:dict[str,dict] = {}
    all_states["module"] = model.state_dict()
    # logger.debug(f"model state dict is {model.state_dict()}")
    all_states["optimizer"] = optimizer.state_dict()

    all_states["primitives"] = {"iteration":iteration}
    with open(out,"wb") as fout:
        torch.save(all_states,fout)

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
    with open(src,"rb") as fin:
        all_states = torch.load(fin)
        model.load_state_dict(all_states["module"])
        # logger.debug(f"model state dict is {all_states["module"]}")
        optimizer.load_state_dict(all_states["optimizer"])
        iteration = all_states["primitives"]["iteration"]
        return iteration











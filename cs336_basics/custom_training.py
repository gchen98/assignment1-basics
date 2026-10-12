import math
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
            logger.debug(f"group {group}")
            self.first_moments
            for p in group["params"]:
                logger.debug(f"parameter {p}")
                params_shape = p.shape
                logger.debug(f"params shape is: {params_shape}")
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

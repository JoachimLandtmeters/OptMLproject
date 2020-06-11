import torch
from torch import Tensor
import math
from Module import Module

class Sequential(Module):
    
    def __init__(self, items ):
        super(Sequential, self).__init__()
        
        # initialise a list of all object 
        self.items = items

    def forward(self, x):
        inp = x
        for item in self.items:
            out = item.forward(inp)
            inp = out
        return out

    def backward(self, grad_output):
        tmp_grad = grad_output
        for item in reversed(self.items):
            tmp_grad = item.backward(tmp_grad)

    def _set_zero(self, x):
        return x.fill_(0)

    def set_zero_grad(self):
        for item in self.items:
            item.set_zero_grad()
            
    def optimisation_step(self, optimizer):
        for item in self.items:
            item.optimisation_step(optimizer)
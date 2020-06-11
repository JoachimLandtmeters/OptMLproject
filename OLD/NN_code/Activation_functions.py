import torch
from torch import Tensor
import math
from Module import Module


################# Activation function definition ###############################
############################# ReLu #############################################

class ReLu(Module):
    def __init__(self):
        self.x = None
        
    def forward(self, x):
        self.x = x.clone()
        x[x < 0] = 0
        return x
        
    def backward(self, grad_output):
        return grad_output.mul(self.x >= 0)
    
############################### Tanh ############################################

class tanh(Module):
    def __init__(self):
        self.x = None
        
    def forward(self, x):
        self.x = x.clone()
        return torch.tanh(x)
        
    def backward(self, grad_output):
        return grad_output.mul(1 - (self.x.tanh().mul(self.x.tanh())))
    
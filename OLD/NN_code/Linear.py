import torch
from torch import Tensor
import math
from Module import Module


class Linear(Module):
    
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        
        # set the size
        self.input_size = input_size
        self.output_size = output_size
        
        # set the weight W 
        self.weight = None
        self.weight_grad = None

        # set the bias b 
        self.bias_grad = None
        self.bias = None
        
        self.weights_initialisation()
        
    def weights_initialisation(self):
        
        # initialise the weight W 
        stdv = 1. / math.sqrt(self.input_size)
        self.weight = Tensor(self.output_size, self.input_size).uniform_(-stdv, stdv)
        self.weight_grad = Tensor(self.output_size, self.input_size)
        
        # initialise the bias b
        stdv = 1. / math.sqrt(self.input_size)
        self.bias = Tensor(self.output_size, 1).uniform_(-stdv, stdv)
        self.bias_grad = Tensor(self.output_size, 1)
    
    
    def forward(self , x):
        # S = W * X + b
        self.x = x.clone()
        return self.weight.mm(self.x).add(self.bias)
    
    def backward(self , grad_output ):
        
        # update the bias gradient db = dL/ds
        self.bias_grad += grad_output.sum(dim=1,keepdim=True)
        
        # update the weights gradient dW =  dL/ds * X^T 
        self.weight_grad += grad_output.mm(self.x.t()) 
        
        # compute the gradient of the output dX = dL/ds * W^T
        return self.weight.t().mm(grad_output)
    
    def param(self):
        return [self.weight, self.bias]

    def param_grad(self):
        return [self.weight_grad, self.bias_grad ]

    def set_zero_grad(self):
        # set the weight W to zero
        self.weight_grad.zero_()
        # set the weight W to zero
        self.bias_grad.zero_()

    def optimisation_step(self, optimizer):
        self.weight = optimizer.optimize(self.weight, self.weight_grad)
        self.bias = optimizer.optimize(self.bias, self.bias_grad)
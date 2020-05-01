from Module import Module
from Sequential import Sequential
from Activation_functions import ReLu, tanh
from Linear import Linear

class Neural_network(Module):
    def __init__(self, hidden_layers):
        Items = []
        linear = Linear(2, 25)
        Items.append(linear)
        Items.append(ReLu())
        for i in range(hidden_layers-1):
            Items.append(Linear(25, 25))
            Items.append(ReLu())
        Items.append(tanh())
        Items.append(Linear(25, 2))
        self.model = Sequential(Items)

    def forward(self, x):
        return self.model.forward(x)

    def backward(self, grad_output):
        return self.model.backward(grad_output)
    
    def set_zero_grad(self):
        self.model.set_zero_grad()

    def optimisation_step(self, optimizer):
        self.model.optimisation_step(optimizer)

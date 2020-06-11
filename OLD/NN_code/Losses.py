import torch
from torch import Tensor
import math
from Module import Module


############################## MSE Loss ##########################################

class MSE(Module):
    
    def __init__(self):
        super(MSE, self).__init__()
        
    def forward(self, y_pred , y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return (y_pred - y_true).pow(2).sum() / y_true.shape[1]
    
    def backward(self):
        return 2*(self.y_pred - self.y_true) / self.y_true.shape[1]
    
    
class Crossentropy(Module):
    def __init__(self):
        super(Crossentropy,self).__init__()
    
    def softmax(self,x):
        """ x is (Num_sample,Num_classe) """
        max_ = torch.max(x, 1, keepdim=True)[0]
        x_exp = torch.exp(x-max_)
        x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
        return x_exp/x_exp_sum
    
    def forward(self, y_pred , y_true):
        """ remarque : here y_true is not one-hot encoded vector """
        self.y_pred = y_pred.t()
        self.y_true = y_true.t()
        self.size_y_true = self.y_true.size(0)
        self.p = self.softmax(self.y_pred)
        log_likelihood = - torch.log(self.p[range(self.size_y_true),self.y_true])
        return log_likelihood.sum()/self.size_y_true
    
    def backward(self):
        grad = self.p
        grad[range(self.size_y_true),self.y_true] -= 1
        return grad.t()/self.size_y_true
class Optimizer(object):
    def optimize (self, weight, weight_grad) :
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def optimize(self, weight, weight_grad) :
        return weight - self.lr*weight_grad
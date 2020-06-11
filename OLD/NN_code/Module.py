
# Mother class for the deep learning framework

class Module ( object ) :
    
    def forward ( self , * x ) :
        raise NotImplementedError
        
    def backward ( self , * grad_output ) :
        raise NotImplementedError
        
    def param ( self ) :
        return []
    
    def param_grad(self):
        pass

    def set_zero_grad(self):
        pass
    
    def weights_initialisation(self):
        pass

    def optimisation_step(self, optimizer):
        pass
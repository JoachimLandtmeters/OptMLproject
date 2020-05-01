import torch
from torch import Tensor

import math
import matplotlib.pyplot as plt

######################### Generate Data ######################################

def generate_disc_data(nb_samples, one_hot_labels = False, normalize = False):
    """ Generate disc data in [-1,1]x[-1,1] with radius = 1/sqrt(2*pi) """

    train_input = Tensor(nb_samples,2).uniform_(-1,1)
    test_input = Tensor(nb_samples,2).uniform_(-1,1)
    
    train_target = (((1/(2*math.pi) - (train_input.pow(2).sum(1))).sign() + 1) / 2).long()
    test_target = (((1/(2*math.pi) - (test_input.pow(2).sum(1))).sign() + 1) / 2).long()
    
    if one_hot_labels:
        train_target = convert_to_one_hot_labels(train_input, train_target)
        test_target = convert_to_one_hot_labels(test_input, test_target)

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)
    
    return train_input, train_target, test_input, test_target
        
        
def convert_to_one_hot_labels(input, target):
    """ Convert target to one hot labels"""
    tmp = input.new_zeros(target.size(0), 2)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

############################## Training ################################################

def train_model(model, train_input, train_target, criterion, optimizer, epochs, mini_batch_size, print_epoch = 1):
    for e in range(epochs):
        sum_loss = 0
        
        for b in range(0, train_input.size(0), mini_batch_size):
            ##Forward propagation
            output = model.forward(train_input.narrow(0, b, mini_batch_size).t())
            
            #Calculate loss
            loss = criterion.forward(output,train_target.narrow(0,b,mini_batch_size).t())

            sum_loss += loss

            # put to zero weights and bias grad
            model.set_zero_grad()

            ##Backpropagation
            #Calculate grad of loss
            loss_grad = criterion.backward()

            #Grad of the model
            model.backward(loss_grad)

            #Update parameters
            model.optimisation_step(optimizer)

        if print_epoch != -1:
            if e % print_epoch == 0 or e == epochs-1:
                print(f"epoch : {e} loss : {sum_loss}")

def evaluate_model(model_class, hidden_layers, nb_samples, criterion, optimizer, epochs, mini_batch_size, validation_ratio, validation_rounds,one_hot_labels_, hide_output):
    
    # Initialize error's list
    train_error = []
    validation_error = []
    test_error = []

    if hide_output:
        print_epoch = -1
    else:
        print_epoch = 25
        
    for val in range(validation_rounds):
        if not hide_output:
            print(f"Evaluation round {val}")

        # Get the data and generate train and validation set
        _input, _target, test_input, test_target = generate_disc_data(nb_samples, one_hot_labels = one_hot_labels_, normalize = True)
        nb_samples_validation = int(validation_ratio*nb_samples)
        validation_input = _input[:nb_samples_validation]
        validation_target = _target[:nb_samples_validation]
        train_input = _input[nb_samples_validation:]
        train_target = _target[nb_samples_validation:]
        
        # Create and train the model
        model = model_class(hidden_layers)
        train_model(model, train_input, train_target, criterion, optimizer, epochs, mini_batch_size, print_epoch)
        
        if not one_hot_labels_:
            def error_computation(model_,input,target):
                return compute_error_not_hot_label(model_, input, target)
        else:
            def error_computation(model_,input,target):
                return compute_error(model_, input, target)
        # Compute different errors
        train_error.append(error_computation(model, train_input, train_target))
        validation_error.append(error_computation(model, validation_input, validation_target))
        test_error.append(error_computation(model, test_input, test_target))
        
        if not hide_output:
            print(f"Train error = {train_error[-1]} Validation error = {validation_error[-1]} Test error = {test_error[-1]}")
        result = [mean(train_error), stdev(train_error), mean(validation_error), stdev(validation_error), mean(test_error), stdev(test_error)]
    return result

        
############################## General use functions ################################################

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def stdev(numbers):
    m = mean(numbers)
    return float((float(sum((xi - m)**2 for xi in numbers)) / len(numbers))**0.5)

def compute_error(model, input, target):
    output = model.forward(input.t())
    _,pred_classes = output.max(0)
    
    target_classes = target[:,-1].long()
    nb_errors = (pred_classes != target_classes).sum()
        
    return (100.0*nb_errors) / input.size(0)

def compute_error_not_hot_label(model, input, target):
    """ note : target must not be one hot labels """
    output = model.forward(input.t())
    _,pred_classes = output.max(0)
   
    nb_errors = (pred_classes != target).sum()
        
    return (100.0*nb_errors) / input.size(0)


############################## Plot functions ################################################

def plot_disc_data(input, target, one_hot_labels = False):
    if one_hot_labels :
        labels = target[:,-1]
    else :
        labels = target
    
    plt.figure(figsize=(8,8))
    plt.scatter(input[:,0] , input[:,1] , s= 1, c = labels)
    plt.show()
    



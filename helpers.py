import torch
from torch import Tensor
from torch import nn
import numpy as np
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

############################## Plot functions ################################################

def plot_disc_data(input, target, one_hot_labels = False):
    if one_hot_labels :
        labels = target[:,-1]
    else :
        labels = target
    
    plt.figure(figsize=(8,8))
    plt.scatter(input[:,0] , input[:,1] , s= 1, c = labels)
    plt.show()
    
def convergence_vizualisation(loss_feedback):
    mean_loss = np.mean(loss_feedback,0)
    std_loss = np.std(loss_feedback,0)
    
    plt.figure(figsize=(10,10))
    plt.plot(mean_loss,label='mean loss')
    plt.plot(mean_loss+2*std_loss,linestyle=':',label='CI up')
    plt.plot(mean_loss-2*std_loss,linestyle=':',label='CI down')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Convergence')
    plt.legend()
    plt.show()


############################# training and testing ###########################################

def train_model(model,epochs,eta,train_input, train_target, batch_size,show_training=True):
    
    optimizer = torch.optim.SGD(model.parameters(), lr = eta)
    criterion = nn.CrossEntropyLoss()
    
    loss_track = []
    for e in range(0, epochs):
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), batch_size):
            output = model(train_input.narrow(0, b, batch_size))
            loss = criterion(output, train_target.narrow(0,b,batch_size))
            sum_loss = sum_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if show_training:
            if e % 25 == 0:
                print(" epochs :",e ," the loss is :",sum_loss)
        loss_track.append(sum_loss)
    return loss_track

def compute_nb_errors(model,test_input, test_target, batch_size) :
    nbr_error = 0
    for b in range(0,test_input.size(0),batch_size):
        output = model(test_input.narrow(0, b, batch_size))
        _, predicted_classes = output.max(1)
        for k in range(batch_size):
            if test_target[b + k] !=predicted_classes[k] :
                nbr_error = nbr_error + 1

    return 100.0*nbr_error/test_input.size(0)

def evaluate_model(model_f, nb_samples, eta, epochs, batch_size, validation_ratio, validation_rounds,one_hot_labels_,show_training=False):
    
    # Initialize error's list
    train_error = []
    validation_error = []
    test_error = []
    loss_feedback = np.zeros((validation_rounds,epochs))
    
    # train and evaluate the model multiple time for better statistics
    for val in range(validation_rounds):
        # Get the data and generate train and validation set
        _input, _target, test_input, test_target = generate_disc_data(nb_samples, one_hot_labels = one_hot_labels_, normalize = True)
        nb_samples_validation = int(validation_ratio*nb_samples)
        validation_input = _input[:nb_samples_validation]
        validation_target = _target[:nb_samples_validation]
        train_input = _input[nb_samples_validation:]
        train_target = _target[nb_samples_validation:]

        # Create and train the model
        model_train = model_f()
        loss_feedback[val,:]=train_model(model_train, epochs,eta,train_input, train_target, batch_size,show_training)

        # Compute different errors
        train_error.append(compute_nb_errors(model_train, train_input, train_target,batch_size))
        validation_error.append(compute_nb_errors(model_train, validation_input, validation_target,batch_size))
        test_error.append(compute_nb_errors(model_train, test_input, test_target,batch_size))


        print(f"Train error = {train_error[-1]} Validation error = {validation_error[-1]} Test error = {test_error[-1]}")
    result = [np.mean(train_error), np.std(train_error), np.mean(validation_error), np.std(validation_error), np.mean(test_error), np.std(test_error)]
    
    return result ,loss_feedback
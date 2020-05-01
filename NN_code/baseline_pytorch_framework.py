import torch
from torch import Tensor
from torch import nn
from helpers import generate_disc_data

def train_model_pytorch(model,epochs,eta,criterion,train_input, train_target, batch_size, print_epoch = 1):
    
    optimizer = torch.optim.SGD(model.parameters(), lr = eta)
    
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
        if print_epoch != -1:
            if e % print_epoch == 0 or e == epochs-1:
                print(f"epoch : {e} loss : {sum_loss}")
        
def compute_nb_errors(model,test_input, test_target, batch_size) :
    nbr_error = 0
    for b in range(0,test_input.size(0),batch_size):
        output = model(test_input.narrow(0, b, batch_size))
        _, predicted_classes = output.max(1)
        for k in range(batch_size):
            if test_target[b + k] !=predicted_classes[k] :
                nbr_error = nbr_error + 1

    return 100.0*nbr_error/test_input.size(0)

def compute_nb_errors_hot_label(model,test_input, test_target, batch_size) :
    nbr_error = 0
    for b in range(0,test_input.size(0),batch_size):
        output = model(test_input.narrow(0, b, batch_size))
        _, predicted_classes = output.max(1)
        for k in range(batch_size):
            if test_target[b + k,-1] !=predicted_classes[k] :
                nbr_error = nbr_error + 1

    return 100.0*nbr_error/test_input.size(0)

def evaluate_model_pytorch(model_f, criterion,eta,nb_samples, epochs, batch_size, validation_ratio, validation_rounds,one_hot_labels_,
                 hide_output):
    
    # Initialize error's list
    train_error = []
    validation_error = []
    test_error = []
    
    if hide_output:
        print_epoch = -1
    else:
        print_epoch = 25
    
    # train and evaluate the model multiple time for better statistics
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
        model_train = model_f()
        train_model_pytorch(model_train, epochs,eta,criterion,train_input, train_target, batch_size, print_epoch)
        
        if one_hot_labels_:
            def error_computation(model_,input,target,batch_size):
                return compute_nb_errors_hot_label(model_, input, target,batch_size)
        else:
            def error_computation(model_,input,target,batch_size):
                return compute_nb_errors(model_, input, target,batch_size)
        
        # Compute different errors
        train_error.append(error_computation(model_train, train_input, train_target,batch_size))
        validation_error.append(error_computation(model_train, validation_input, validation_target,batch_size))
        test_error.append(error_computation(model_train, test_input, test_target,batch_size))


        print(f"Train error = {train_error[-1]} Validation error = {validation_error[-1]} Test error = {test_error[-1]}")
    result = [np.mean(train_error), np.std(train_error), np.mean(validation_error), np.std(validation_error), np.mean(test_error), np.std(test_error)]
    
    return result
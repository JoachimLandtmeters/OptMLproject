import torch
from torch import Tensor
from torch import nn
import numpy as np
import math
import matplotlib.pyplot as plt

from time import time




"""
Model building functions
"""
def fully_connected_NN(sizes):
    layers = []
    for l in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[l],sizes[l+1]))
        if l<len(sizes)-2:
            layers.append(nn.ReLU())
        else:
            layers.append(nn.LogSoftmax(dim=1))
            
    FCNN = nn.Sequential(*layers)
    return FCNN




"""
Optimization functions
"""
def optimize(optimizer, epochs, trainloader, valloader, model, criterion , method = None ):
    train_losses = []
    test_losses = []
    accuracy = []
    time0 = time()
    
    for e in range(epochs):
        print("Epoch {}".format(e))
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            def closure():
                # Training pass
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                #This is where the model learns by backpropagating, not needed for linesearch
                if loss.requires_grad:
                    loss.backward()
                return loss

            def closure_hf():
                # Training pass
                
                output = model(images)
                loss = criterion(output, labels)
                #This is where the model learns by backpropagating, not needed for linesearch
                if loss.requires_grad:
                    loss.backward(create_graph=True)
                return loss , output
            
            #And optimizes its weights here
            if method== "LBFGS" :
                optimizer.step(closure)
                
            elif method == "HF" :
                optimizer.zero_grad()
                optimizer.step(closure_hf, M_inv=None)
                
            else :
                closure()
                optimizer.step()
                
                
            with torch.no_grad():
                output = model(images)
                loss_ = criterion(output, labels)
            running_loss += loss_.item()

            
        # Compute the test loss
        # we let torch know that we dont intend to call .backward
        
        print("Training loss: {}".format(running_loss/len(trainloader)))
        train_losses.append( running_loss/len(trainloader))
        
        acc = accuracy_test(valloader, model)
        accuracy.append(acc)

        test_loss = 0
        with torch.no_grad():
            model.eval()
            for test_images, test_labels in valloader:

                test_images = test_images.view(test_images.shape[0], -1)

                test_output = model(test_images)
                tloss = criterion(test_output, test_labels)

                test_loss += tloss.item()


        print("Test loss: {}".format(test_loss/len(valloader)),"\n")
        test_losses.append(test_loss/len(valloader))


    print("\nTraining Time (in minutes) =",(time()-time0)/60)
    training_time = (time()-time0)/60

    return test_losses, train_losses, accuracy



"""
Prediction and performance evaluation functions
"""
def predict_one_img(valloader,model):
    
    images, labels = next(iter(valloader))
    img = images[0].view(1, 784)
    with torch.no_grad():
        logps = model(img)
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    
    predictions = probab.index(max(probab))
    print("Probabilities computed for each digit =\n",probab)
    print("\nPredicted Digit =", predictions)
    print("Actual Digit =",labels[0].numpy())
    #view_classify(img.view(1, 28, 28), ps)
    
    return probab, predictions

def accuracy_test(valloader, model):
    correct_count, all_count = 0, 0
    for images,labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
    
    accuracy = correct_count/all_count
    print("Number Of Images Tested =", all_count)
    print("Model Accuracy =", accuracy)
    
    return accuracy
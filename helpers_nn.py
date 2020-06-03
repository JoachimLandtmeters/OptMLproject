import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math
import matplotlib.pyplot as plt

from time import time

"""
This file contains multiple sections, each containing functions tied to the title section:

A: Model building
B: Optimization
C: Prediction and performance evaluation
D: Hyperparameter tuning

"""



"""
Model building functions
"""

# Fully connected neural network
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


# Convolutional neural network (two convolutional layers)
# Taken from adventuresinML GitHub
class ConvNet(nn.Module):
    def __init__(self,image_size):
        super(ConvNet, self).__init__()
        # First convolutional layer is defined.
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Second convolutional layer is defined.
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        # Fully connected layers are defined
        self.fc1 = nn.Linear(int(image_size/4) * int(image_size/4) * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    #this method overrides the forward method in nn.Module
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out




"""
Optimization functions
"""

# CNN optimization
def optimize_CNN(optimizer, epochs, trainloader, valloader, model, criterion, method = None ):
    train_losses = []
    test_losses = []
    accuracy = []
    time0 = time()

    for e in range(epochs):
        print("Epoch {}".format(e))
        running_loss = 0
        for images, labels in trainloader:
            def closure_sd():
                 optimizer.zero_grad()
                 output = model(images)
                 loss = criterion(output, labels)
                 loss.backward()
                 return loss

            if method == "SdLBFGS":
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                #This is where the model learns by backpropagating
                loss.backward()
                optimizer.step(closure_sd)
                running_loss += loss.item()

            else:
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                #This is where the model learns by backpropagating
                loss.backward()#And optimizes its weights here
                optimizer.step()
                running_loss += loss.item()



             # Compute the test loss
            # we let torch know that we dont intend to call .backward

        print("Training loss: {}".format(running_loss/len(trainloader)))
        train_losses.append( running_loss/len(trainloader))

        acc = accuracy_test_CNN(valloader, model)
        accuracy.append(acc)

        test_loss = 0
        with torch.no_grad():
            model.eval()
            for test_images, test_labels in valloader:

                test_output = model(test_images)
                tloss = criterion(test_output, test_labels)

                test_loss += tloss.item()


        print("Test loss: {}".format(test_loss/len(valloader)),"\n")
        test_losses.append(test_loss/len(valloader))


    print("\nTraining Time (in minutes) =",(time()-time0)/60)
    training_time = (time()-time0)/60

    return train_losses, test_losses


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

            def closure_sd():
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                return loss



            #And optimizes its weights here
            if method== "LBFGS" :
                optimizer.step(closure)

            elif method == "HF" :
                optimizer.zero_grad()
                optimizer.step(closure_hf, M_inv=None)

            elif method == "SdLBFGS" :
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step(closure_sd)

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

    return test_losses, train_losses



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

def accuracy_test_CNN(valloader, model):
    correct_count, all_count = 0, 0
    model.eval()
    for images,labels in valloader:
        with torch.no_grad():
            outputs = model(images)
            _,pred_labels=torch.max(outputs.data, 1)
            correct_count+=(pred_labels==labels).sum().item()
            all_count += labels.size(0)

    accuracy = correct_count/all_count
    print("Number Of Images Tested =", all_count)
    print("Model Accuracy =", accuracy)

    return accuracy



"""
Hyperparameter tuning functions
"""
def best_learning_rate(input_size, output_size, trainloader, valloader):
    best_loss = None
    best_learning_rate = None

    grid = 0.1 * 2**np.arange(5)
    print("Learning rates to try:", grid)

    for lr in grid:
        model = create_model(input_size, output_size)
        criterion = nn.NLLLoss()
        momentum=0.9
        optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

        epochs = 15
        test_losses, train_losses, accuracy = optimize(optimizer, epochs, trainloader, valloader, lr, momentum, model, criterion)
        best_loss_achieved = np.min(test_losses)
        plt.scatter(lr, best_loss_achieved)
        if best_loss is None or best_loss_achieved < best_loss:
            best_loss = best_loss_achieved
            best_learning_rate = lr
    return best_learning_rate

## Optimization for machine learning project - EPFL, AY 2019/20
#### Authors: Margherita Guido, Daniele Hamm, Pierre Vuillecard

In this folder we collected the code used to produce results for the project of
EPFL Optimization for machine learning course.
It consists of second order optimization methods applied to Fully Connected and Convolutional
Neural Networks.

All the code is implemented in Python using Pytorch, Torchvision and Numpy packages.
The Datasets used are "Digit MNIST" and "Fashion MNIST", both available in
Torchvision package and loaded in the beginning of every notebook used.

Here follows a brief description of what this folder consists of.
The work is subdivided in four folders:

- ***Main***

  This folder contains:

  - ``main.ipynb``.
    This notebook has been created to show the content of both datasets and how they are loaded; moreover it provides an effective example of how the helper functions are used to create the two different Neural Network architectures and to train them for a given method. The displayed output corresponds to an SGD run (our baseline method) on both architectures used, for the Digit MNIST dataset.

  - ``helper.py`` contains all the functions used for optimization.
  The remaining files ``curveball.py`` (https://github.com/jotaf98/pytorch-curveball), ``sdlbfgs.py`` (https://github.com/harryliew/SdLBFGS) and ``lbfgsnew.py`` (https://github.com/SarodYatawatta/federated-pytorch-test/tree/master/src) contain the methods that we used that are not already implemented in Pytorch package. The source is previously indicated.


- ***Hyperparameters_Tuning***

  This folder contains all the results that collected from the training of all
  the methods for both datasets and different sets of parameters, to select the ones considered in final comparison, as described in the report.

  - The results are subdivided in  folders as follows:
    - There are two folders corresponding to the two datasets **MNIST** and **FashionMNIST**
    - For each dataset there are two folders corresponding to the two different models used **FCNN** and **CNN**
    - In each of the above folders, there is a subfolder for each of the different methods that have been used (**SGD**,**LBFGSFullbatch**, **LBFGSMinibatch**, **LBFGSNew**, **SdLBFGS** and **CurveBall**).

    The *Hyperparameters_Tuning/Data/Mod/Meth* folder contains then all the results obtained training model *Mod* on dataset *Data* using the optimization method *Meth*.

  The results consist of in ``.txt``  files that contain train and test accuracy, train and test loss and training time for each epoch of the training, and for every different combination of parameters.
    Where it was needed, a notebook named ``plot_results.ipynb`` is present. It helped to analyze the results in order to select the best parameters. Its output is saved in the folder **best_results**.

  - The code used to produce the results is contained  in the folder **Notebooks**.
  Inside here the ``.ipynb`` notebooks follow the same naming convention as before, indicating method, dataset and model used.
  For practical reasons (parallel running of different notebooks), there is one notebook for each method and architecture.
  The file ``helper.py`` contains all the functions used for optimization.
    The remaining files ``curveball.py`` (LINK), ``sdlbfgs.py`` (LINK) and ``lbfgsnew.py`` (LINK) contain the methods that we used that are not already implemented in Pytorch package. The source is previously indicated.



- ***Methods_Comparison***

  With the same convention as above, the folder *Methods Comparison/Data/Mod* contains the comparison of different second order methods (applied with selected best parameters) for the training of model *Mod* on dataset *Data*.
  Inside here, the notebook ``Data_Mod.ipynb`` contains the code to compare train and test accuracy and train loss of methods with best parameters, whose performance data are stored inside the corresponding ``.txt`` files.
  The outputs of the notebook are saved in the folder *Images*.



- ***Overfit_Analysis***  

  Here the Notebook and results used to analyse the overfitting phenomenon in the training of the Neural Networks are contained.
  The data used are stored in ``.txt`` files, the output images produced are saved in the folder **images**.
  The notebook used to produce the plots is named ``Overfit_analysis.ipynb``.

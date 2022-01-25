<!-- <h2> <i> Citation </i> </h2> -->
<i> <b> If you like our work and find useful, please cite as: </b> </i> <br>
```
@article{chowdhury2021deepqgho,
  title={DeepQGHO: Quantized Greedy Hyperparameter Optimization in Deep Neural Networks for on-the-fly Learning}, <br>
  author={Chowdhury, Anjir Ahmed and Hossen, Md Abir and Azam, Md Ali and Rahman, Md Hafizur}, <br>
  year={2021}
}
```

# GHO : Greedy Approch Based Hyperpameter Optimization
```greedy algorithm``` ```deep learning``` ```neural networks``` ```online learning``` ```hyperparameter optimization``` ```edge device``` ```Jetson Nano```
<p align="center" width="25%"><img width="25%" src="GHO.png"></p>
<h2> About the project </h2>
Hyperparameter optimization or tuning plays a significant role in the performance and reliability of deep learning (DL). Many hyperparameter optimization algorithms have been developed for obtaining better validation accuracy in DL training. Most state-of-the-art hyperparameters are computationally expensive due to a focus on validation accuracy. Therefore, they are unsuitable for online or on-the-fly training applications which require computational efficiency. In this project, we develop a novel greedy approach-based hyperparameter optimization (GHO) algorithm for faster training applications, e.g., on-the-fly training..
<h3> Built With </h3>
<p>
List of frameworks and packages
  <li><a href="https://www.python.org/downloads/release/python-360/">Python 3.7</a>
  <li><a href="https://www.tensorflow.org/">Tensorflow 2.4</a>
  <li><a href="https://keras.io/keras_tuner/">KerasTuner</a>
  <li><a href="https://jupyter.org/">Jupyter Notebook</a>
</p>
<h2> Project Description </h2>
We performed eight experiments on two different platforms (PC and Edge), each based on a different DNN  architecture and dataset (CIFAR10 and the Intel Image Classification). We  specified the same hyper-parameter configuration space to fairly compare RandomSearch, Bayesian optimization and GHO. The optimal hyper-parameter configuration was determined by each of the HPO algorithms based on the highest validation accuracy. To prevent the time complexity, we fixed the activation function: relu, optimizer: adam, epochs: 100, kernel size: (3, 3). pool size: (2, 2), and stride size: (2, 2). We also use the early stopping callback function to prevent the overfitting of the model and define, monitor: 'val_loss', and  patience : 3.

<b> Test Cases on PC: </b> All the testcases for PC are found in [main_test_case](https://github.com/abirhossen786/greedyhpo/tree/main/main/Main_test_case) under the name of each dataset. Under each dataset there are jupyter files where all the codes for different HPO algorithms on different architecture are given. Already generated results are generated on a machine with a 32-core AMD Ryzen Threadripper TR4 processor, NVIDIA RTX A6000 GPU card, 48 GB of GPU memory, and 128 GB of DDR4 CPU memory.

<b> Test Cases on Edge Platform: </b> For edge platform we use [Nvidia Jetson Nano (2GB)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/education-projects/). All the python codes are found in [JetsonNano_test_case](https://github.com/abirhossen786/greedyhpo/tree/main/main/JetsonNano_test_case). Along with that there are screenshots which record the device status to justify our findings. 
<h3> Datasets </h3>
  <li><a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR10</a>
  <li><a href="https://www.kaggle.com/puneet6060/intel-image-classification">Intel Image Classification</a>
<h3> Architectures </h3>
    for PC,
    <li> Vgg16 </li>
    <li> ResNet50 </li>
    for NVIDIA Jetson Nano,
    <li> Vgg11 </li>
    <li> ResNet18 </li>
    
<h3> HPO Algorithms </h3>
  <li><a href="https://www.jmlr.org/papers/v13/bergstra12a.html?source=post_page---------------------------">Random Search</a>
  <li><a href="https://papers.nips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf">Bayesian Optimization</a>
  <li> GHO (ours) </li>
<h2> How to use </h2>
<h3> Random Search and Bayesian Optimization:</h3> 
For implementing standard HPO algorithm we use KerasTuner API. Click <a href="https://keras.io/keras_tuner/">here</a> to get detail implementation guide for implementing the algorithms. Just download the datasets and follow the step by step process of the guideline to successfully implemented the algorithms.
<h3> GHO Algorithm:</h3>
<b> Step 1: </b> Download or Import dataset along with the necessary Package

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,MaxPool2D, Activation, Flatten,Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam

```
<b> Step 2: </b> Define the search space

```python
layer_sizes_1 = [16,32,64,128,256,512]
layer_sizes_2 = [16,32,64,128,256,512]
layer_sizes_3 = [16,32,64,128,256,512]
layer_sizes_4 = [16,32,64,128,256,512]
layer_sizes_5 = [16,32,64,128,256,512]
dense_sizes_1 = [64,128,256,512]
dense_sizes_2 = [64,128,256,512]
learning_rates_1 = [1e-3,1e-1,1e-2,1e-4, 1e-5]
dp_sizes_1 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
dp_sizes_2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
dp_sizes_3 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
dp_sizes_4 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
dp_sizes_5 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
dp_sizes_6 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
wd_sizes_1=[0,0.1, 0.001, 0.0001]
```

<b> Step 3: </b> Define the default values for each hyperparameters

```python
opt_layer_1=16
opt_layer_2=16
opt_layer_3=16
opt_layer_4=16
opt_layer_5=16
opt_layer_6=0.001
opt_layer_7=64
opt_layer_8=64
opt_layer_9=0.0
opt_layer_10=0.0
opt_layer_11=0.0
opt_layer_12=0.0
opt_layer_13=0.0
opt_layer_14=0.0
opt_layer_15=0.0
```
<!---<h2> Contacts </h2>--->
<!--- 
```
Anjir Ahmed Chowdhury
Department of Computer Science, Primeasia University, Bangladesh
anjir.ahmed@primeasia.edu.bd
```
```
Md Ali Azam 
Department of Electrical Engineering, South Dakota School of Mines and Technology, Rapid City, SD 57701 USA 
azam.ete.ruet@gmail.com
```
```
Md Abir Hossen
Computer Scicence and Engineering Department, University of South Carolina, SC 29208 USA 
mhossen@email.sc.edu
```
--->



# GHO : Greedy Approch Based Hyperpameter Optimization
<p align="center" width="25%"><img width="25%" src="GHO.png"></p>
<h2> About the project </h2>
Hyperparameter optimization or tuning plays a significant role in the performance and reliability of deep learning (DL). Many hyperparameter optimization algorithms have been developed for obtaining better validation accuracy in DL training. Most state-of-the-art hyperparameters are computationally expensive due to a focus on validation accuracy. Therefore, they are unsuitable for online or on-the-fly training applications which require computational efficiency. In this project, we develop a novel greedy approach-based hyperparameter optimization (GHO) algorithm for faster training applications, e.g., on-the-fly training.
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
    for PC we selected,
    <li> Vgg16 </li>
    <li> ResNet50 </li>
    for Edge Platform,
    <li> Vgg11 </li>
    <li> ResNet18 </li>

<h2> How to use </h2>
<h3> Random Search and Bayesian Optimization:</h3> 
For implementing standard HPO algorithm we use KerasTuner API. Click <a href="https://keras.io/keras_tuner/">here</a> to get detail implementation guide for implementing the algorithms. Just download the datasets and follow the step by step process of the guideline to successfully implemented the algorithms.
<h3> GHO Algorithm:</h3>

<h2> Citation </h2>

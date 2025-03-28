# Adaptive-SGD
This repository contains the code for the numerical part in our paper [*Adaptive Stochastic Gradient Descents on Manifolds with an Application on Weighted Low-Rank Approximation*](https://arxiv.org/abs/2503.11833). 
The project is implemented by [Peiqi Yang](https://pqyang9.github.io/) and Conglong Xu from Georege Washington Universitly Mathematics Department, advised by [Prof. Hao Wu](https://sites.google.com/site/haowugwu/home).

# Repository Structure

[AdaSGD.py](AdaSGD.py): Implementation of stochastic gradient descent with adaptive learning rates. (Algorithm 3.20)  
[DetSGD.py](DetSGD.py): Implementation of stochastic gradient descent with deterministic learning rates for comparison. (Algorithm 3.13 in [XYW25](https://arxiv.org/abs/2502.14174))  
[weights-sampling.py](weights-sampling.py): Sampling weights and generating stochastic iterates for algorithms.  
[test-asgd.py](test-asgd.py): Numerical experiment of stochastic gradient descent with adaptive learning rates.  
[test-dsgd.py](test-dsgd.py): Numerical experiment of stochastic gradient descent with deterministic learning rates.  
[figures.py](figures.py): Generating the figure of stochastic gradient descent with adaptive or deterministic learning rates under different parameters.  
[figures-sep.py](figures-sep.py): Separating the figure in three scales for comparison.

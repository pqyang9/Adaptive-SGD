import numpy as np
import pickle
from matplotlib import pyplot as plt
import AdaSGD  # import the algorithms

# open the pickle file storing the 'list_of_index_netflix' in binary mode
with open('pickles/list_of_index_netflix.pkl', 'rb') as file:
    # deserialize the results
    list_of_index_netflix = pickle.load(file)

# set up the test data
w = np.load('matrix/w-weights.npy')         # load the weights matrix
a = np.load('matrix/netflix-data.npy')      # load the movie rating matrix generated from weights-sampling.py
m = len(w)                                  # number of rows
n = len(w[0])                               # number of columns

iteration_numbers = 1000
k = 32

# set up the parameters for adaptive steps
alpha = 10**5
epsilon = 0.5**6


# set up parameters for each case
lmbda = 1/(10**4)
kappa = 1/(10**7)

# run the adaptive stochastic gradient descent with alpha
iterations, time_elapsed_list, params, grads, costs, costs_unreg = ada.optimize_iterations_stochastic(a, w, k, lmbda, iteration_numbers, list_of_index_netflix, kappa, alpha, epsilon)

# create a list to store the output from the experiment
adaptive_sgd_stiefel_netflix = [iterations, time_elapsed_list, params, grads, costs, costs_unreg]
# Open a file and use dump()
with open('pickles/adaptive_sgd_stiefel_netflix_l4_k7.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(adaptive_sgd_stiefel_netflix, file)


print("finished")

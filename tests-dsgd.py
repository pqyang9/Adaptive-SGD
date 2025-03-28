import numpy as np
import pickle
from matplotlib import pyplot as plt
import DetSGD # import the algorithms

# open the pickle file storing the 'list_of_index_netflix' in binary mode
with open('pickles/list_of_index_netflix.pkl', 'rb') as file:
    # deserialize the results
    list_of_index_netflix = pickle.load(file)

# set up the test data for confined sgd
w = np.load('matrix/w-weights.npy')   # load the weights matrix
a = np.load('matrix/netflix-data.npy')    # load the movie rating matrix
m = len(w)      # number of rows
n = len(w[0])       # number of columns

iteration_numbers = 1000
k = 32
lmbda = 1/(10**8)
log_K = 4

# run the confined stochastic gradient descent
iterations, time_elapsed_list, params, grads, costs, costs_unreg = csgd.optimize_iterations_stochastic(a, w, k, lmbda, iteration_numbers, list_of_index_netflix, log_K)
print("number of iterations is: ", iterations)
print("time elapsed is: ", time_elapsed_list[-1])


# create a list to store the output from the experiment
confined_sgd_stiefel_netflix = [iterations, time_elapsed_list, params, grads, costs, costs_unreg]
# Open a file and use dump()
with open('pickles/confined_sgd_stiefel_netflix_l8_k4.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(confined_sgd_stiefel_netflix, file)


print("finished")
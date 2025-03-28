import numpy as np
import pickle

w = np.load('matrix/w-weights.npy')   # load the weights matrix
m = len(w)      # number of rows
n = len(w[0])       # number of columns

num_of_ones = 0     # number of non-zero entries
for i in range(m):
    for j in range(n):
        if w[i][j] > 0:
            num_of_ones = num_of_ones + 1
print("m = ", m, "n = ", n, "number of ones =", num_of_ones)

# store the indices of all non-zero entries
list_ones_index = [(i, j) for i in range(m) for j in range(n) if w[i][j] != 0]

# set random seed
np.random.seed(1234)

# generate the sampled indices
number_of_iterations = 1100
list_sampled_index = np.random.randint(low=0, high=num_of_ones, size=number_of_iterations)
list_of_index_netflix = []
for i in list_sampled_index:
    list_of_index_netflix.append(list_ones_index[i])
print(len(list_of_index_netflix))    # checkpoint

# store the sampled indices for further use
with open('pickles/list_of_index_netflix.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(list_of_index_netflix, file)

print("finished")
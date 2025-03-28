import pickle
from matplotlib import pyplot as plt

# open the pickle files storing the adaptive stochastic gradient descent with Stiefel manifolds
with open('pickles/adaptive_sgd_stiefel_netflix_l4_ka7.pkl', 'rb') as file:
    # deserialize the output
    vars_list_ada_stiefel_netflix_1 = pickle.load(file)
with open('pickles/adaptive_sgd_stiefel_netflix_l6_ka_5en9.pkl', 'rb') as file:
    # deserialize the output
    vars_list_ada_stiefel_netflix_2 = pickle.load(file)
with open('pickles/adaptive_sgd_stiefel_netflix_l8_ka_5en11.pkl', 'rb') as file:
    # deserialize the output
    vars_list_ada_stiefel_netflix_3 = pickle.load(file)

# open the pickle files storing the confined (deterministic) stochastic gradient descent with Stiefel manifolds
with open('pickles/confined_sgd_stiefel_netflix_l4_k4.pkl', 'rb') as file:
    # deserialize the output
    vars_list_csgd_stiefel_netflix_1 = pickle.load(file)
with open('pickles/confined_sgd_stiefel_netflix_l6_k4.pkl', 'rb') as file:
    # deserialize the output
    vars_list_csgd_stiefel_netflix_2 = pickle.load(file)
with open('pickles/confined_sgd_stiefel_netflix_l8_k4.pkl', 'rb') as file:
    # deserialize the output
    vars_list_csgd_stiefel_netflix_3 = pickle.load(file)


# retrieve the costs functions with regularization removed
costs_unreg_ada_stiefel_netflix_1 = vars_list_ada_stiefel_netflix_1[5]
costs_unreg_ada_stiefel_netflix_2 = vars_list_ada_stiefel_netflix_2[5]
costs_unreg_ada_stiefel_netflix_3 = vars_list_ada_stiefel_netflix_3[5]
costs_unreg_csgd_stiefel_netflix_1 = vars_list_csgd_stiefel_netflix_1[5]
costs_unreg_csgd_stiefel_netflix_2 = vars_list_csgd_stiefel_netflix_2[5]
costs_unreg_csgd_stiefel_netflix_3 = vars_list_csgd_stiefel_netflix_3[5]


# figure: comparing the convergence speed, adaptive vs. confined sgd
fig = plt.figure(figsize=(15, 5), constrained_layout=True)
fig.suptitle('SGD with Adaptive vs Deterministic Learning Rates for Different Lambda')

# create a frame of 1-by-3 subfigs
subfigs = fig.subfigures(nrows=1, ncols=3, wspace=0.07, width_ratios=[1., 1., 1.])
# plot the subfig for lambda=10^-4
axs0 = subfigs[0].subplots(1)
subfigs[0].suptitle(r'(a) $\lambda = 10^{-4}$, $\kappa = 10^{-7}$')
axs0.plot(costs_unreg_ada_stiefel_netflix_1, color='red', label='Adaptive SGD')
axs0.plot(costs_unreg_csgd_stiefel_netflix_1, color='blue', label='Deterministic SGD')
axs0.set_ylabel('cost functions with regularization removed')
axs0.set_xlabel('log(t): log of number of iterations')
axs0.set_xscale('log')
axs0.legend()
# plot the subfig for lambda=10^-6
axs1 = subfigs[1].subplots(1)
subfigs[1].suptitle(r'(b) $\lambda = 10^{-6}$, $\kappa = 5 \times 10^{-9}$')
axs1.plot(costs_unreg_ada_stiefel_netflix_2, color='red', label='Adaptive SGD')
axs1.plot(costs_unreg_csgd_stiefel_netflix_2, color='blue', label='Deterministic SGD')
axs1.set_ylabel('cost functions with regularization removed')
axs1.set_xlabel('log(t): log of number of iterations')
axs1.set_xscale('log')
axs1.legend()
# plot the subfig for lambda=10^-8
axs2 = subfigs[2].subplots(1)
subfigs[2].suptitle(r'(c) $\lambda = 10^{-8}$, $\kappa = 5 \times 10^{-11}$')
axs2.plot(costs_unreg_ada_stiefel_netflix_3, color='red', label='Adaptive SGD')
axs2.plot(costs_unreg_csgd_stiefel_netflix_3, color='blue', label='Deterministic SGD')
axs2.set_ylabel('cost function with regularization removed')
axs2.set_xlabel('log(t): log of number of iterations')
axs2.set_xscale('log')
axs2.legend()
# save the figure
plt.savefig('figures/Adaptive-vs-Confined-SGD-Sep.png')

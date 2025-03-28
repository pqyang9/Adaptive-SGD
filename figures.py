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


# figure: comparing the convergence speed of adaptive and confined with different lambda
fig = plt.figure(figsize=(6, 6), constrained_layout=True)
fig.suptitle('SGD with Adaptive vs Deterministic Learning Rates')

axs0 = fig.subplots(1)
axs0.plot(costs_unreg_ada_stiefel_netflix_1, color='purple', label=r'Adaptive: $\lambda = 10^{-4}$, $\kappa = 10^{-7}$')
axs0.plot(costs_unreg_ada_stiefel_netflix_2, color='cyan', label=r'Adaptive: $\lambda = 10^{-6}$, $\kappa = 5 \times 10^{-9}$')
axs0.plot(costs_unreg_ada_stiefel_netflix_3, color='yellow', label=r'Adaptive: $\lambda = 10^{-8}$, $\kappa = 5 \times 10^{-11}$')
axs0.plot(costs_unreg_csgd_stiefel_netflix_1, color='brown', label=r'Deterministic: $\lambda = 10^{-4}$')
axs0.plot(costs_unreg_csgd_stiefel_netflix_2, color='violet', label=r'Deterministic: $\lambda = 10^{-6}$')
axs0.plot(costs_unreg_csgd_stiefel_netflix_3, color='green', label=r'Deterministic: $\lambda = 10^{-8}$')
axs0.set_ylabel('cost functions with regularization removed')
axs0.set_xlabel('log(t): log of number of iterations')
axs0.set_xscale('log')
axs0.legend()
# save the figure
plt.savefig('figures/Adaptive-vs-Confined-SGD.png')


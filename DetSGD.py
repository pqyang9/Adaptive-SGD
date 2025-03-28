import numpy as np
import copy
import time
import math

def cost_regularized(a, w, u, s, vh, lmbda):
    # Inputs:
    # a: target matrix [m x n]
    # w: weight matrix [m x n]
    # u: left multiplying matrix in SVD [m x k]
    # vh: right multiplying matrix in SVD [k x n]
    # s: vector of singular values in SVD [k]
    # lmbda: regularization parameter (scalar)
    # Output:
    # the value of the cost function in Problem 3.2

    s = np.diag(s)
    p = np.dot(np.dot(u, s), vh)
    cost = np.sum(w*((a - p)**2)) + lmbda*np.linalg.norm(p, 'fro')**2
    return cost


def cost_unregularized(a, w, u, s, vh):
    # Inputs:
    # a: target matrix [m x n]
    # w: weight matrix [m x n]
    # u: left multiplying matrix in SVD [m x k]
    # vh: right multiplying matrix in SVD [k x n]
    # s: vector of singular values in SVD [k]
    # Output:
    # the value of the cost function in Problem 4.1

    s = np.diag(s)
    p = np.dot(np.dot(u, s), vh)
    cost = np.sum(w*((a - p)**2))
    return cost


# use reduced SVD to generate Eckart-Young initialization
def initial(a, k):  # k = the rank constraint
    # Inputs:
    # a: target matrix [m x n]
    # k: low rank constraint (scalar)
    # Outputs:
    # u: Eckart-Young solution in u direction [m x k]
    # vh: Eckart-Young solution in v direction [k x n]
    # s: Eckart-Young solution in s direction [k]

    # Reduced SVD of A
    u, s, vh = np.linalg.svd(a, full_matrices=False)

    # Generate the Eckart-Young solution
    u = u[:, : k]
    s = s[: k]
    vh = vh[: k, :]

    return u, s, vh


# QR decomposition -- the retraction map on Stiefel manifold
def qr(x):
    # Input:
    # x: matrix for QR decomposition [m x k]
    # Output:
    # q: orthogonal matrix Q from QR decomposition [m x k]

    if len(x) >= len(x[0]):
        q, r = np.linalg.qr(x)
        for i in range(len(r)): # scan for the negative diagonal entries in R
            if r[i][i] < 0:
                q[:,i] = -q[:,i]
    else:   # take transpose of flat matrix
        q, r = np.linalg.qr(np.transpose(x))
        for i in range(len(r)): # scan for the negative diagonal entries in R
            if r[i][i] < 0:
                q[:,i] = -q[:,i]
        q = np.transpose(q)
    return q


# project tangent vector of Euclidean space to the tangent space of Stiefel (sub)manifold
def projector(x, z):
    # Inputs:
    # x: the point on the Stiefel manifold [m x n]
    # z: the tangent vector in the tangent space at x [m x n]
    # Output:
    # project z to the tangent space of Stiefel (sub)manifold at x -- formula in A.1

    m = np.matmul(np.transpose(x), z)
    symmetric = (m + np.transpose(m))/2
    projected = z - np.matmul(x, symmetric)
    return projected


# compute the gradient for confined SGD
def stochastic_gradient(a, w, u, s, vh, lmbda, list_of_index, iterations):
    # Inputs:
    # a: target matrix [m x n]
    # w: weight matrix [m x n]
    # u: left multiplying matrix in SVD [m x k]
    # vh: right multiplying matrix in SVD [k x n]
    # s: vector of singular values in SVD [k]
    # lmbda: regularization parameter (scalar)
    # list_of_index: list of the sampled indices for SGD (list(200))
    # iterations: the number of iteration (scalar)
    # Outputs:
    # du: the gradient on u direction in formula (A.1) [m x k]
    # dvh: the gradient on v direction in formula (A.1) [k x n]
    # ds: the gradient on s direction in formula (A.1) [k]
    # the cost function value with regularization at the input point (u, s, vh)
    # the cost function value with regularization removed at the input point (u, s, vh)

    # compute the cost functions
    cost = cost_regularized(a, w, u, s, vh, lmbda)
    cost_unreg = cost_unregularized(a, w, u, s, vh)

    # retrieve the sampled entry
    (eta, gamma) = list_of_index[iterations]
    a_eta_gamma = a[eta][gamma]
    p_eta_gamma = np.dot((s*u[eta]), vh[:, gamma])
    diff_eta_gamma = a_eta_gamma - p_eta_gamma

    # update the gradients
    gradu = np.zeros_like(u)
    gradu[eta] = -2*diff_eta_gamma*(s*vh[:,gamma])
    gradvh = np.zeros_like(vh)
    gradvh[:, gamma] = -2*diff_eta_gamma*(s*u[eta])
    grads = -2*diff_eta_gamma*(u[eta]*vh[:, gamma])

    # project the gradients
    du = projector(u, gradu)
    dvh = projector(vh, gradvh)
    ds = grads + 2*lmbda*s

    grads ={"du": du,
            "ds": ds,
            "dvh": dvh}
    return grads, cost, cost_unreg


def step_with_confinement(a, k, lmbda, s_initial, phi_min_modifier):
    # Inputs:
    # a: target matrix [m x n]
    # k: low rank constraint (scalar)
    # lmbda: regularization parameter (scalar)
    # s_initial: the initial value of the vector component of the updating step [k]
    # phi_min_modifier: a scalar multiplied to the step size to accelerate the convergence speed and enhance the stability (scalar)
    # Output:
    # phi_step: step size with the confinement (scalar)

    # set up the parameters
    alpha = np.max(np.square(a))
    s_norm = np.linalg.norm(s_initial)
    rho_0 = max(s_norm**2, alpha/(4*lmbda))     # Equation 3.8 in SGD paper

    # compute phi-min (Equation 3.10 in SGD paper)
    phi_min = max((lmbda + 2*math.sqrt(lmbda) + 1)*alpha, math.sqrt(32*k*alpha*lmbda + 8*k*(2 + lmbda*lmbda)*(2*lmbda*rho_0 + (math.pi**2 + 12)/6)))

    # adjust phi-min using modifier
    phi_step = phi_min*(10**phi_min_modifier)

    return phi_step


def optimize_iterations_stochastic(a, w, k, lmbda, iteration_numbers, list_of_index, phi_min_modifier):
    # Inputs:
    # a: target matrix [m x n]
    # w: weight matrix [m x n]
    # k: low rank constraint (scalar)
    # lmbda: regularization parameter (scalar)
    # iteration_numbers: number of descent iterations (integer)
    # phi_min_modifier: a scalar multiplied to the step size to accelerate the convergence speed and reduce the stability (scalar)
    # Outputs:
    # iterations: total number of descent steps (integer)
    # time_elapsed: total time used on the descent iterations (scalar)
    # params: last iterate step (dictionary)
    # grads: gradient in the last iterate step (dictionary)
    # costs: the value of cost function along the descent process (list)

    # set up the cost function recorder
    costs = []
    costs_unreg = []
    iterations = 0

    # generate initial values: 'initial_func' controls the initialization method
    u, s, vh = initial(a, k)
    u = copy.deepcopy(u)
    s = copy.deepcopy(s)
    vh = copy.deepcopy(vh)

    # generate step-size modifier
    phi_step = step_with_confinement(a, k, lmbda, s, phi_min_modifier)

    # compute descent steps
    # set up the runtime recorder
    time_elapsed_list = []
    # record the initial time
    time_start = time.time()

    while iterations <= iteration_numbers:
        # computing the gradients and cost function values
        grads, cost, cost_unreg = stochastic_gradient(a, w, u, s, vh, lmbda, list_of_index, iterations)

        # retrieve the gradients
        du = grads["du"]
        ds = grads["ds"]
        dvh = grads["dvh"]

        # computing the step size with confinement
        iterations = iterations + 1
        step = (1/iterations)/phi_step

        # updating the step
        u = qr(u - step * du)
        s = s - step * ds
        vh = qr(vh - step * dvh)

        # computing the runtime
        time_stamp = time.time()
        time_elapsed = time_stamp - time_start

        # record the costs, unregularized costs, and runtime
        costs.append(cost)
        costs_unreg.append(cost_unreg)
        time_elapsed_list.append(time_elapsed)

    params = {"u": u,
              "s": s,
              "vh": vh}

    grads = {"du": du,
             "ds": ds,
             "dvh": dvh}

    return iterations, time_elapsed_list, params, grads, costs, costs_unreg
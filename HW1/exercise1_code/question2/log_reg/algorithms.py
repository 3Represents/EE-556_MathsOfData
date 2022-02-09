import time
import numpy as np
# import scipy.sparse.linalg as spla
from numpy.random import randint
# from scipy.sparse.linalg.dsolve import linsolve
import numpy.linalg as LA

from log_reg.utils import print_end_message, print_start_message, print_progress


##########################################################################
# Unconstrained methods
##########################################################################

def GD(fx, gradf, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Gradient Descent'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    x, maxit = parameter['x0'], parameter['maxit']
    alpha = 1 / parameter['Lips']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit,'x': np.zeros([maxit, x.shape[0]])}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = x - alpha * gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)
        info['x'][iter] = x

        # Print the information.
        if (iter %  5 ==0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    info['iter'] = maxit
    print_end_message(method_name, time.time() - tic_start)
    return x, info


# Gradient with strong convexity
def GDstr(fx, gradf, parameter) :
    """
    Function:  GDstr(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    method_name = 'Gradient Descent with strong convexity'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    x, maxit = parameter['x0'], parameter['maxit']
    alpha = 2 / (parameter['Lips'] + parameter['strcnvx'])
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit,'x': np.zeros([maxit, x.shape[0]])}

    # Main loop.
    for iter in range(maxit):
        # Start timer
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = x - alpha * gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)
        info['x'][iter] = x
        
        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# Accelerated gradient
def AGD(fx, gradf, parameter):
    """
    Function:  AGD (fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx	  - strong convexity parameter
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
    x = y = parameter['x0']
    t = 1
    alpha = 1 / parameter['Lips']
    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = y - alpha * gradf(y)
        t_next = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y_next = x_next + (t - 1) / t_next * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x, t, y = x_next, t_next, y_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# Accelerated gradient with strong convexity
def AGDstr(fx, gradf, parameter):
    """
    Function:  AGDstr(fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx	  - strong convexity parameter
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    method_name = 'Accelerated Gradient with strong convexity'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
    x = y = parameter['x0']
    alpha = 1 / parameter['Lips']
    sq_L, sq_mu = np.sqrt(parameter['Lips']), np.sqrt(parameter['strcnvx'])
    t = (sq_L - sq_mu) / (sq_L + sq_mu)
    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = y - alpha * gradf(y)
        y_next = x_next + t * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x, y = x_next, y_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# LSGD
def LSGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSGD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent with line-search.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    method_name = 'Gradient Descent with line search'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
    x, L, maxit = parameter['x0'], parameter['Lips'], parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        d_k = -gradf(x)
        L_0 = L / 2

        i = 0
        while fx(x + d_k / (2 ** i * L_0)) > fx(x) - LA.norm(d_k) ** 2 / (2 ** (i + 1) * L_0):
            i += 1

        L = 2 ** i * L_0
        x_next = x + d_k / L

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# LSAGD
def LSAGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGD (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient with line search'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
    x = y = parameter['x0']
    t, L, maxit = 1, parameter['Lips'], parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        d_k = -gradf(y)
        L_0 = L / 2

        i = 0
        while fx(y + d_k / (2 ** i * L_0)) > fx(y) - LA.norm(d_k) ** 2 / (2 ** (i + 1) * L_0):
            i += 1

        L_next = 2 ** i * L_0
        x_next = y + d_k / L_next
        t_next = (1 + np.sqrt(1 + 4 * L_next / L * t ** 2)) / 2
        y_next = x_next + (t - 1) / t_next * (x_next - x) 

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x, y, t, L = x_next, y_next, t_next, L_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# AGDR
def AGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = AGDR (fx, gradf, parameter)
    Purpose:   Implementation of the AGD with adaptive restart.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient with restart'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y, t and find the initial function value (fval).
    x = y = parameter['x0']
    t = 1
    alpha = 1 / parameter['Lips']
    fval, maxit = fx(x), parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = y - alpha * gradf(y)
        fval_next = fx(x_next)
        
        if fval < fval_next:
            y, t = x, 1
            x_next = y - alpha * gradf(y)
            fval_next = fx(x_next)

        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2    
        y_next = x_next + (t - 1) * (x_next - x) / t_next

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x, t, y, fval = x_next, t_next, y_next, fval_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# LSAGDR
def LSAGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGDR (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search and adaptive restart.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient with line search + restart'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y, t and find the initial function value (fval).
    x = y = parameter['x0']
    t, fval, L, maxit = 1, fx(x), parameter['Lips'], parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    line_search = lambda i: \
        i if fx(y + d_k / (2 ** i * L_0)) <= fx(y) - LA.norm(d_k) ** 2 / (2 ** (i + 1) * L_0) \
        else line_search(i+1)

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        d_k = -gradf(y)
        L_0 = L / 2

        L_next = 2 ** line_search(0) * L_0
        x_next = y + d_k / L_next
        fval_next = fx(x_next)

        if fval < fval_next:
            y, t = x, 1
            d_k = -gradf(y)
            L_next = 2 ** line_search(0) * L_0
            x_next = y + d_k / L_next
            fval_next = fx(x_next)

        t_next = (1 + np.sqrt(1 + 4 * L_next / L * t ** 2)) / 2
        y_next = x_next + (t - 1) / t_next * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x, y, t, L, fval = x_next, y_next, t_next, L_next, fval_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


def AdaGrad(fx, gradf, parameter):
    """
    Function:  [x, info] = AdaGrad (fx, gradf, hessf, parameter)
    Purpose:   Implementation of the adaptive gradient method with scalar step-size.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Adaptive Gradient method'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, B0, alpha, grad (and any other)
    x, maxit = parameter['x0'], parameter['maxit']
    Q, alpha, grad, delta = 0, 1, gradf(x), 1e-5
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        Q_next = Q + LA.norm(grad) ** 2
        H = (np.sqrt(Q_next) + delta) * np.identity(len(x))
        x_next = x - alpha * LA.inv(H) @ grad

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x, Q, grad = x_next, Q_next, gradf(x)

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# Newton
def ADAM(fx, gradf, parameter):
    """
    Function:  [x, info] = ADAM (fx, gradf, parameter)
    Purpose:   Implementation of ADAM.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'ADAM'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, beta1, beta2, alpha, epsilon (and any other)
    x, maxit = parameter['x0'], parameter['maxit']
    beta1, beta2, alpha, epsilon = 0.9, 0.999, 0.1, 1e-8
    m = v = 0
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()
        k = iter + 1

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        g = gradf(x)
        m_next = beta1 * m + (1 - beta1) * g
        v_next = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m_next / (1 - beta1 ** k)
        v_hat = v_next / (1 - beta2 ** k)
        H = np.sqrt(v_hat) + epsilon
        x_next = x - alpha * m_hat / H

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x, m, v = x_next, m_next, v_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


def SGD(fx, gradfsto, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradfsto:
    :param parameter:
    :return:
    """
    method_name = 'Stochastic Gradient Descent'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    x, maxit, k = parameter['x0'], parameter['maxit'], 1
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        i = randint(parameter['no0functions'])
        x_next = x - gradfsto(x, i) / k

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        k += 1

    print_end_message(method_name, time.time() - tic_start)
    return x, info


def SAG(fx, gradfsto, parameter):
    """
    Function:  [x, info] = SAG(fx, gradfsto, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
    :param fx:
    :param gradfsto:
    :param parameter:
    :return:
    """
    method_name = 'Stochastic Gradient Descent with averaging'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    x, n, maxit = parameter['x0'], parameter['no0functions'], parameter['maxit']
    alpha = 1 / (16 * parameter['Lmax'] * n)
    v = np.zeros([n, len(x)])
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        i = randint(n)
        v[i] = gradfsto(x, i)
        x_next = x - alpha * v.sum(axis=0)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


def SVR(fx, gradf, gradfsto, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param gradfsto:
    :param parameter:
    :return:
    """
    method_name = 'Stochastic Gradient Descent with variance reduction'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    x, n, Lmax, maxit = parameter['x0'], parameter['no0functions'], parameter['Lmax'], parameter['maxit']
    gamma, q = 0.01 / Lmax, int(1e3 * Lmax)
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_l = x_tilde = x
        v = gradf(x_tilde)
        X = np.zeros([q, len(x)])
        
        for l in range(q):
            i = randint(n)
            v_l = gradfsto(x_l, i) - gradfsto(x_tilde, i) + v
            xl_next = x_l - gamma * v_l
            X[l] = xl_next
            x_l = xl_next
        
        x_next = X.sum(axis=0) / q

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


##########################################################################
# Prox
##########################################################################

def SubG(fx, gx, gradfx, parameter):
    """
    Function:  [x, info] = subgrad(fx, gx, gradfx, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
    :param fx:
    :param gx:
    :param gradfx:
    :param parameter:
    :return:
    """
    method_name = 'Subgradient'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize lambda, maxit, x0, alpha and subgrad function
    G = 54  # pre computed with np.linalg.norm(A)
    R = 0.41529129 # pre computed with np.linalg.norm(x0 - xstar)
    lmbd, maxit, x0 = parameter['lambda'], parameter['maxit'], parameter['x0']
    subgrad = lambda x: gradfx(x) + lmbd * np.sign(x)
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop
    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update the next iteration (x and alpha)
        alpha = R / (G * np.sqrt(k+1))
        x_next = x_k - alpha * subgrad(x_k)

        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % parameter['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

        x_k = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x_k, info


def ista(fx, gx, gradf, proxg, params):
    """
    Function:  [x, info] = ista(fx, gx, gradf, proxg, parameter)
    Purpose:   Implementation of ISTA.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               prox_Lips  - Lipschitz constant for gradient.
               lambda     - regularization factor in F(x)=f(x)+lambda*g(x).
    :param fx:
    :param gx:
    :param gradf:
    :param proxg:
    :param parameter:
    :return:
    """
    method_name = 'ISTA'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize parameters.
    lmbd, maxit, x0 = params['lambda'], params['maxit'], params['x0']
    alpha = 1 / params['prox_Lips']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update the iterate
        x_next = proxg(x_k - alpha * gradf(x_k), alpha * lmbd)

        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % params['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

        x_k = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x_k, info


def fista(fx, gx, gradf, proxg, params):
    """
    Function:  [x, info] = fista(fx, gx, gradf, proxg, parameter)
    Purpose:   Implementation of FISTA (with optional restart).
    Parameter: x0            - Initial estimate.
               maxit         - Maximum number of iterations.
               prox_Lips     - Lipschitz constant for gradient.
               lambda        - regularization factor in F(x)=f(x)+lambda*g(x).
               restart_fista - enable restart.
    :param fx:
    :param gx:
    :param gradf:
    :param proxg:
    :param parameter:
    :return:
    """
    if params['restart_fista']:
        method_name = 'FISTAR'
    else:
        method_name = 'FISTA'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize parameters
    lmbd, maxit, x0 = params['lambda'], params['maxit'], params['x0']
    y_k, t_k, alpha = x0, 1, 1 / params['prox_Lips']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update iterate
        x_next = proxg(y_k - alpha * gradf(y_k), alpha * lmbd)
        t_next = (1 + np.sqrt(4 * t_k ** 2 + 1)) / 2

        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % params['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

        if params['restart_fista'] and gradient_scheme_restart_condition(x_k, x_next, y_k):
            t_k = t_next = 1
            y_k = x_k
            x_next = proxg(y_k - alpha * gradf(y_k), alpha * lmbd)

        y_next = x_next + (t_k - 1) / t_next * (x_next - x_k)
        x_k, t_k, y_k = x_next, t_next, y_next

    print_end_message(method_name, time.time() - tic_start)
    return x_k, info


def gradient_scheme_restart_condition(x_k, x_k_next, y_k):
    """
    Whether to restart
    """
    return (y_k - x_k_next) @ (x_k_next - x_k) > 0


def prox_sg(fx, gx, gradfsto, proxg, params):
    """
    Function:  [x, info] = prox_sg(fx, gx, gradfsto, proxg, parameter)
    Purpose:   Implementation of ISTA.
    Parameter: x0                - Initial estimate.
               maxit             - Maximum number of iterations.
               prox_Lips         - Lipschitz constant for gradient.
               lambda            - regularization factor in F(x)=f(x)+lambda*g(x).
               no0functions      - number of elements in the finite sum in the objective.
               stoch_rate_regime - step size as a function of the iterate k.
    :param fx:
    :param gx:
    :param gradfsto:
    :param proxg:
    :param parameter:
    :return:
    """
    method_name = 'PROXSG'
    print_start_message(method_name)

    tic_start = time.time()

    # Initialize parameters
    lmbd, maxit, x0, gamma = params['lambda'], params['maxit'], params['x0'], params['stoch_rate_regime']
    n, X_avg, X_acc, gamma_acc = params['no0functions'], None, np.zeros_like(x0), 0.
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update the average iterate
        gamma_k = gamma(k)
        gamma_acc += gamma_k
        X_acc += gamma_k * x_k
        X_avg = X_acc / gamma_acc
        i = randint(n)
        x_next = proxg(x_k - gamma_k * gradfsto(x_k, i), gamma_k * lmbd)

        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(X_avg) + lmbd * gx(X_avg)
        if k % params['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(X_avg), gx(X_avg))

        x_k = x_next

    print_end_message(method_name, time.time() - tic_start)
    return X_avg, info

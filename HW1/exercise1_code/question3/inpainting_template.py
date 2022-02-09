import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity as ssim # compare_ssim deprecated

from common.utils import *
from common.operators import TV_norm, Representation_Operator, p_omega, p_omega_t, l1_prox



def ISTA(fx, gx, gradf, proxg, params):
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


def FISTA(fx, gx, gradf, proxg, params, verbose=False):
    """
    Function:  [x, info] = fista(fx, gx, gradf, proxg, parameter)
    Purpose:   Implementation of FISTA (with optional restart).
    Parameter: x0            - Initial estimate.
               maxit         - Maximum number of iterations.
               prox_Lips     - Lipschitz constant for gradient.
               lambda        - regularization factor in F(x)=f(x)+lambda*g(x).
               restart_criterion - enable restart.
    :param fx:
    :param gx:
    :param gradf:
    :param proxg:
    :param parameter:
    :return:
    """
    if params['restart_criterion']:
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
        if verbose and k % params['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

        if params['restart_criterion'] and gradient_scheme_restart_condition(x_k, x_next, y_k):
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
    return (y_k - x_k_next).T @ (x_k_next - x_k) > 0


def reconstructL1(image, indices, optimizer, params):
    # Wavelet operator
    r = Representation_Operator(m=params["m"])

    # Define the overall operator
    forward_operator = lambda x: p_omega(r.WT(x), indices) # P_Omega.W^T
    adjoint_operator = lambda x: r.W(p_omega_t(x, indices, params['m'])) # W.P_Omega^T

    # Generate measurements
    b = p_omega(image.reshape(-1), indices)

    fx = lambda x: 0.5 * np.linalg.norm(b - forward_operator(x)) ** 2
    gx = lambda x: params['lambda'] * np.linalg.norm(x, 1)
    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)
    gradf = lambda x: adjoint_operator(forward_operator(x) - b)

    x, info = optimizer(fx, gx, gradf, proxg, params)
    return r.WT(x).reshape((params['m'], params['m'])), info


def reconstructTV(image, indices, optimizer, params):
    """
        image: undersampled image (mxm) to be reconstructed
        indices: indices of the undersampled locations
        optimizer: method of reconstruction (FISTA/ISTA function handle)
        params:
    """
    # Define the overall operator
    forward_operator = lambda x: p_omega(x, indices) # P_Omega
    adjoint_operator = lambda x: p_omega_t(x, indices, params['m'])  # P_Omega^T

    # Generate measurements
    b = forward_operator(image.reshape(-1))

    fx = lambda x: 0.5 * np.linalg.norm(b - forward_operator(x)) ** 2
    gx = lambda x: params['lambda'] * TV_norm(x)
    proxg = lambda x, y: denoise_tv_chambolle(x.reshape((params['m'], params['m'])),
                                              weight=params["lambda"] * y, eps=1e-5,
                                              n_iter_max=50).reshape((params['N'], 1))
    gradf = lambda x: adjoint_operator(forward_operator(x) - b).reshape((params['N'], 1))

    x, info = optimizer(fx, gx, gradf, proxg, params)
    return x.reshape((params['m'], params['m'])), info


# %%

if __name__ == "__main__":

    ##############################
    # Load image and sample mask #
    ##############################
    shape = (256, 256)
    params = {
        'maxit': 200,
        'tol': 10e-15,
        'prox_Lips': 1,
        'lambda': 0.05,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        'restart_criterion': True,
        'stopping_criterion': None,
        'iter_print': 50,
        'shape': shape,
        'restart_param': 50,
        'verbose': True,
        'm': shape[0],
        'rate': 0.4,
        'N': shape[0] * shape[1]
    }
    PATH = './data/2008.jpg'
    image = load_image(PATH, params['shape'])

    im_us, mask = apply_random_mask(image, params['rate'])
    indices = np.nonzero(mask.flatten())[0]
    params['indices'] = indices
    # Choose optimization parameters


    #######################################
    # Reconstruction with L1 and TV norms #
    #######################################
    t_start = time.time()
    reconstruction_l1 = reconstructL1(image, indices, FISTA, params)[0]
    reconstruction_l1 = reconstruction_l1.astype(image.dtype)
    t_l1 = time.time() - t_start

    psnr_l1 = psnr(image, reconstruction_l1)
    ssim_l1 = ssim(image, reconstruction_l1)

    t_start = time.time()
    reconstruction_tv = reconstructTV(image, indices, FISTA, params)[0]
    reconstruction_tv = reconstruction_tv.astype(image.dtype)
    t_tv = time.time() - t_start

    psnr_tv = psnr(image, reconstruction_tv)
    ssim_tv = ssim(image, reconstruction_tv)

    # Plot the reconstructed image alongside the original image and PSNR
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(im_us, cmap='gray')
    ax[1].set_title('Original with missing pixels')
    ax[2].imshow(reconstruction_l1, cmap="gray")
    ax[2].set_title('L1 - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_l1, ssim_l1, t_l1))
    ax[3].imshow(reconstruction_tv, cmap="gray")
    ax[3].set_title('TV - PSNR = {:.2f}\n SSIM  = {:.2f}  - Time: {:.2f}s'.format(psnr_tv, ssim_tv, t_tv))
    [axi.set_axis_off() for axi in ax.flatten()]
    plt.tight_layout()
    plt.savefig('./results.pdf', bbox_inches='tight')
    plt.show()

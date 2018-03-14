# Copyright (C) 2018 Zhixian MA <zxma_sjtu@qq.com>

"""
Utilities for Gaussian mixture model
"""

import numpy as np

def gmm_gen_1d(num_gauss, alpha, theta, numsample=1):
    """Generate gaussian mixture model
    
    inputs
    ======
    num_gauss: int
        Number of gaussian models
    alpha: np.ndarray
        coefficients of the components,
        whose shape is [num_gauss, 1]
    theta: list or tuple or np.ndarray
        parameters w.r.t. GMMs, shape is [2*num_gauss,1]
    numsample: int
        number of samples to be generated
    
    outputs
    =======
    y: float
        The GMM distributed observation state
        
    reference
    =========
    https://stats.stackexchange.com/questions/70855/
    generating-random-variables-from-a-mixture-of-normal-distributions
    """
    # y = np.zeros([numsample,])
    # normalize alpha
    alpha = np.array(alpha)
    if alpha.sum() != 1.0:
        alpha = alpha / alpha.sum()
    
    alpha_cum = np.cumsum(alpha)
    
    # uniformly generate numsample numbers
    # np.random.seed(5201314)
    y_uni = np.random.uniform(0.0, 1.0, [numsample,1])
    y_idx = np.sum((y_uni - alpha_cum) > 0, axis=1)
    # get gamma
    gamma = np.zeros((numsample,num_gauss), dtype=float)
    for i in range(num_gauss):
        gamma[y_idx == i, i] = 1.0
        
    # generate
    mu = np.array(theta[0:-1:2])
    sigma = np.array(theta[1::2])
    if len(mu) != len(sigma):
        print("Paramters are missing..")
        return
    else:
        y_gauss = sigma * np.random.randn(numsample, num_gauss) + mu
        y = np.sum(y_gauss*gamma, axis=1)

    return y


def gauss_fun(y, alpha, mu, sigma):
    """Generate gaussian function"""
    prob = alpha * 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(y - mu)**2 / (2*sigma**2))
    return prob


def step_expectation(obs_data, num_gauss, alpha, mu, sigma):
    """The expectation step"""
    num_samples = len(obs_data)
    gamma = np.zeros([num_samples, num_gauss], dtype=float)
    # calc probs
    # alpha = alpha/alpha.sum()
    for i in range(num_gauss):
        gamma[:,i] = gauss_fun(obs_data, alpha[i], mu[i], sigma[i])
    # calc final gamma
    gamma_sum = np.repeat(np.sum(gamma, axis=1), num_gauss).reshape(gamma.shape)
    gamma = gamma / gamma_sum
        
    return gamma
    

def step_maximization(obs_data, num_gauss, gamma):
    """The maximization step"""
    alpha = np.zeros(num_gauss, dtype=float)
    mu = np.zeros(num_gauss, dtype=float)
    sigma = np.zeros(num_gauss, dtype=float)
    # Estimate params
    for i in range(num_gauss):
        mu[i] = np.sum(gamma[:,i]*obs_data) / np.sum(gamma[:,i])
        sigma[i] = np.sqrt(np.sum(gamma[:,i]*(obs_data - mu[i])**2) / np.sum(gamma[:,i]))
        alpha[i] = np.sum(gamma[:,i]) / len(obs_data)
        
    return alpha/alpha.sum(),mu,sigma


def gmm_est_1d(obs_data, num_gauss=3, init_param=None, num_iter=100, min_dist=1e-10,print_flag=False):
    """Estimation of GMM paramters with EM algorithm
    
    inputs
    ======
    obs_data: np.ndarray
        The observed data
    num_gauss: int
        number of gaussion components
    init_param: list or tuple or np.ndarray
        initialized parameters, default as None
    num_iter: int
        iteration times, default as 100
    min_dist: float
        minimum distances between paramters for two neighbored iterations
    
    output
    ======
    theta: np.ndarray
        Estimated parameters
    """
    
    # Init
    theta = np.zeros([num_gauss*2, ])
    if init_param is None:
        # np.random.seed(5201314)
        alpha = np.ones(num_gauss, dtype=float) / num_gauss
        mu = np.random.uniform(0,1,num_gauss)
        sigma = np.random.uniform(0,1,num_gauss)
    else:
        alpha = init_param[0:-2:3]
        mu = init_param[1:-1:3]
        sigma = init_param[2::3]
        if len(alpha) != len(mu) or len(alpha) != len(sigma) or len(mu) != len(sigma):
            print("Paramters are non-equal lengths..")
            return None
    
    print("Initialized parameters")
    print("alpha:", alpha)
    print("mu   :", mu)
    print("sigma:", sigma)
    
    # Estimate params iteratively
    alpha_last = alpha
    mu_last = mu
    sigma_last = sigma
    for it in range(num_iter):
        # E step
        gamma = step_expectation(obs_data, num_gauss, alpha, mu, sigma)
        # M step
        alpha,mu,sigma = step_maximization(obs_data, num_gauss, gamma)
        # Calc distance
        alpha_dist = np.abs(alpha-alpha_last)
        mu_dist = np.abs(mu-mu_last)
        sigma_dist = np.abs(sigma-sigma_last)
        # update
        alpha_last = alpha
        mu_last = mu
        sigma_last = sigma
        # print 
        if (it+1) % 5 == 0 and print_flag:
            print("Iteration: %d" % (it+1))
            print("alphas: %s  alpha_dists: %s" % (" ".join(alpha.astype(str)), " ".join(alpha_dist.astype(str))))
            print("mus   : %s  mu_dists   : %s" % (" ".join(mu.astype(str)), " ".join(mu_dist.astype(str))))
            print("sigmas: %s  sigma_dists: %s" % (" ".join(sigma.astype(str)), " ".join(sigma_dist.astype(str))))
        # stop
        max_d = np.vstack([alpha_dist, mu_dist, sigma_dist]).max()
        if max_d <= min_dist:
            break
            
    theta[0:-1:2] = mu
    theta[1::2] = sigma
    
    return alpha/alpha.sum(),theta


def gmm_gen_2d(num_gauss, alpha, theta, numsample=1):
    """Generate gaussian mixture model 2D case
    
    inputs
    ======
    num_gauss: int
        Number of gaussian models
    alpha: np.ndarray
        coefficients of the components,
        whose shape is [num_gauss, 1]
    theta: list or tuple or np.ndarray
        parameters w.r.t. GMMs, shape is [2*num_gauss,2]
    numsample: int
        number of samples to be generated
    
    outputs
    =======
    y: float
        The GMM distributed observation state
        
    reference
    =========
    https://stats.stackexchange.com/questions/70855/
    generating-random-variables-from-a-mixture-of-normal-distributions
    """
    # y = np.zeros([numsample,])
    # normalize alpha
    alpha = np.array(alpha)
    if alpha.sum() != 1.0:
        alpha = alpha / alpha.sum()
    
    alpha_cum = np.cumsum(alpha)
    
    # uniformly generate numsample numbers
    np.random.seed(5201314)
    y_uni = np.random.uniform(0.0, 1.0, [numsample,1])
    y_idx = np.sum((y_uni - alpha_cum) > 0, axis=1)
    # get gamma
    gamma = np.zeros((numsample,num_gauss), dtype=float)
    for i in range(num_gauss):
        gamma[y_idx == i, i] = 1.0
        
    # generate
    mu1 = np.array(theta[0, 0:-1:2])
    mu2 = np.array(theta[1, 0:-1:2])
    sigma1 = np.array(theta[0, 1::2])
    sigma2 = np.array(theta[1, 1::2])
    if len(mu1) != len(sigma1):
        print("Paramters are missing..")
        return
    else:
        y_gauss_d1 = sigma1 * np.random.randn(numsample, num_gauss) + mu1
        y_gauss_d2 = sigma2 * np.random.randn(numsample, num_gauss) + mu2
        y_d1 = np.sum(y_gauss_d1*gamma, axis=1)
        y_d2 = np.sum(y_gauss_d2*gamma, axis=1)

    return y_d1,y_d2


def gauss_fun_2d(y, alpha, mu1, sigma1, mu2, sigma2):
    """Generate gaussian function"""
    prob1 = 1/(np.sqrt(2*np.pi)*sigma1)*np.exp(-(y[:,0] - mu1)**2 / (2*sigma1**2))
    prob2 = 1/(np.sqrt(2*np.pi)*sigma2)*np.exp(-(y[:,1] - mu2)**2 / (2*sigma2**2))
    prob = alpha*prob1 * prob2
    return prob


def step_expectation_2d(obs_data, num_gauss, alpha, mu1, sigma1, mu2, sigma2):
    """The expectation step"""
    num_samples = obs_data.shape[0]
    gamma = np.zeros([num_samples, num_gauss], dtype=float)
    # calc probs
    # alpha = alpha/alpha.sum()
    for i in range(num_gauss):
        gamma[:,i] = gauss_fun_2d(obs_data, alpha[i], 
                                  mu1[i], sigma1[i],
                                  mu2[i], sigma2[i])
    # calc final gamma
    gamma_sum = np.repeat(np.sum(gamma, axis=1), num_gauss).reshape(gamma.shape)
    gamma = gamma / gamma_sum
        
    return gamma
    

def step_maximization_2d(obs_data, num_gauss, gamma):
    """The maximization step"""
    alpha = np.zeros(num_gauss, dtype=float)
    mu1 = np.zeros(num_gauss, dtype=float)
    mu2 = np.zeros(num_gauss, dtype=float)
    sigma1 = np.zeros(num_gauss, dtype=float)
    sigma2 = np.zeros(num_gauss, dtype=float)
    # Estimate params
    for i in range(num_gauss):
        mu1[i] = np.sum(gamma[:,i]*obs_data[:,0]) / np.sum(gamma[:,i])
        mu2[i] = np.sum(gamma[:,i]*obs_data[:,1]) / np.sum(gamma[:,i])
        sigma1[i] = np.sqrt(np.sum(gamma[:,i]*(obs_data[:,0] - mu1[i])**2) / np.sum(gamma[:,i]))
        sigma2[i] = np.sqrt(np.sum(gamma[:,i]*(obs_data[:,1] - mu2[i])**2) / np.sum(gamma[:,i]))
        alpha[i] = np.sum(gamma[:,i]) / obs_data.shape[0]
        
    return alpha/alpha.sum(),mu1,sigma1,mu2,sigma2


def gmm_est_2d(obs_data, num_gauss=3, init_param=None, num_iter=100, min_dist=1e-10):
    """Estimation of GMM paramters with EM algorithm 2D simple case
    
    inputs
    ======
    obs_data: np.ndarray
        The observed data
    num_gauss: int
        number of gaussion components
    init_param: dict
        initialized parameters, default as None
    num_iter: int
        iteration times, default as 100
    min_dist: float
        minimum distances between paramters for two neighbored iterations
    
    output
    ======
    alpha: np.ndarray
        Estimated GMM coefficients
    theta: np.ndarray
        Estimated parameters
    """
    
    # Init
    theta = np.zeros([2, num_gauss*2])
    if init_param is None:
        # np.random.seed(5201314)
        alpha = np.ones(num_gauss, dtype=float) / num_gauss
        mu1 = np.random.uniform(0,1,num_gauss)
        mu2 = np.random.uniform(0,1,num_gauss)
        sigma1 = np.random.uniform(0,1,num_gauss)
        sigma2 = np.random.uniform(0,1,num_gauss)
    else:
        alpha = init_param["alpha"]
        mu1 = init_param["theta"][0,0:-1:2]
        mu2 = init_param["theta"][1,0:-1:2]
        sigma1 = init_param["theta"][0, 1::2]
        sigma2 = init_param["theta"][1, 1::2]
        if len(alpha) != len(mu1) or len(alpha) != len(sigma1) or len(mu1) != len(sigma1):
            print("Paramters are non-equal lengths..")
            return None
    
    print("Initialized parameters")
    print("alpha:", alpha)
    print("mu1   :", mu1)
    print("sigma1:", sigma1)
    print("mu2   :", mu2)
    print("sigma2:", sigma2)
        
    # Estimate params iteratively
    alpha_last = alpha
    mu_last1 = mu1
    mu_last2 = mu2
    sigma_last1 = sigma1
    sigma_last2 = sigma2
    for it in range(num_iter):
        # E step
        gamma = step_expectation_2d(
            obs_data, num_gauss, alpha, 
            mu1, sigma1, mu2, sigma2)
        # M step
        alpha,mu1,sigma1,mu2,sigma2 = step_maximization_2d(
            obs_data, num_gauss, gamma)
        # Calc distance
        alpha_dist = np.abs(alpha-alpha_last)
        mu_dist1 = np.abs(mu1-mu_last1)
        mu_dist2 = np.abs(mu2-mu_last2)
        sigma_dist1 = np.abs(sigma1-sigma_last1)
        sigma_dist2 = np.abs(sigma2-sigma_last2)
        # update
        alpha_last = alpha
        mu_last1 = mu1
        mu_last2 = mu2
        sigma_last1 = sigma1
        sigma_last2 = sigma2
        
        # stop
        max_d = np.vstack([alpha_dist, mu_dist1, mu_dist2, sigma_dist1, sigma_dist2]).max()
        if max_d <= min_dist:
            break
            
    theta[0, 0:-1:2] = mu1
    theta[1, 0:-1:2] = mu2
    theta[0, 1::2] = sigma1
    theta[1, 1::2] = sigma2
    
    return alpha/alpha.sum(),theta
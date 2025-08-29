import numpy as np

def invlogit(y):
  return 1 / (1 + np.exp(-y))

def logit(x):
  return np.log(x) - np.log1p(-x)

def pool_par_gauss(alpha, m, v):
    '''
    Obtain the parameters of the pooled distribution (Gaussian).

    :param alpha: numpy array of weights, which must lie in a simplex.
    :param m: numpy array of means, same dimension as 'alpha' and 'v'.
    :param v: numpy array vector of *variances*.
    :return: pooled mean and *STANDARD DEVIATION*.

    references: See url{https://github.com/maxbiostat/logPoolR}
    '''
    ws = alpha/v
    vstar = 1/sum(ws)
    mstar = sum(ws*m) * vstar
    return mstar, np.sqrt(vstar)

def alpha_01(alpha_inv):
    '''
    Maps from R^n to the open simplex.

    :param alpha_inv: a numpy array in R^n
    :return: a numpy array on the (n+1) open simplex.

    references: See url{https://github.com/maxbiostat/logPoolR}
    '''

    K = len(alpha_inv) + 1
    z = np.full(K - 1, np.nan)
    alphas = np.zeros(K)

    for k in range(1, K):
        z[k-1] = invlogit(alpha_inv[k-1] + np.log( 1 / (K - k) ))
        alphas[k-1] = (1 - sum(alphas[0:(k-1)])) * z[k-1]

    alphas[K-1] = 1-sum(alphas[0:K])

    return alphas

def alpha_real(alpha):
    '''
    Maps a (n+1) open simplex to R^n.

    :param alpha: numpy array of weights that live on an open simplex.
    :return: a projection of 'alpha' onto R^n.

    references: See url{https://github.com/maxbiostat/logPoolR}
    '''
    p = len(alpha)
    if(p == 1):
        return logit(alpha)
    else:
        return np.log(alpha[0:(p-1)] / alpha[p-1])

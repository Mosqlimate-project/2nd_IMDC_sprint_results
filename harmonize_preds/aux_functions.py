import numpy as np
from numpy import log
from numpy import exp
from scipy.stats import lognorm
from scipy.optimize import minimize

import harmonize_preds.pyLogPool

def quantile_pair(p):
    return ((1-p)/2, (1+p)/2)

def get_lognormal_pars2(
    med: float,
    lwr: float,
    upr: float,
    conf_level: float = 0.90,
    fn_loss: str = "median",
) -> tuple:
    """
    Estimate the parameters of a log-normal distribution based on forecasted median,
    lower, and upper bounds.

    This function estimates the mu and sigma parameters of a log-normal distribution
    given a forecast's known median (`med`), lower (`lwr`), and upper (`upr`) confidence
    interval bounds. The optimization minimizes the discrepancy between the theoretical
    quantiles of the log-normal distribution and the provided forecast values.

    Parameters
    ----------
    med : float
        The median of the forecast distribution.
    lwr : float
        The lower bound of the forecast (corresponding to `(1 - alpha)/2` quantile).
    upr : float
        The upper bound of the forecast (corresponding to `(1 + alpha)/2` quantile).
    Conf_level : float, optional, default=0.90
        Confidence level used to define the lower and upper bounds.
    fn_loss : {'median', 'lower'}, optional, default='median'
        The optimization criterion for fitting the log-normal distribution:
        - 'median': Minimizes the error in estimating `med` and `upr`.
        - 'lower': Minimizes the error in estimating `lwr` and `upr`.

    Returns
    -------
    tuple
        A tuple `(mu, sigma)`, where:
        - `mu` is the estimated location parameter of the log-normal distribution.
        - `sigma` is the estimated scale parameter.

    Notes
    -----
    - The function uses the Nelder-Mead optimization method to minimize the loss function.
    - If `fn_loss='median'`, the optimization prioritizes minimizing the difference
      between the estimated and actual median (`med`) and upper bound (`upr`).
    - If `fn_loss='lower'`, the optimization prioritizes minimizing the difference
      between the estimated lower bound (`lwr`) and upper bound (`upr`).
    """

    if fn_loss not in {"median", "lower"}:
        raise ValueError(
            "Invalid value for fn_loss. Choose 'median' or 'lower'."
        )

    if any(x < 0 for x in [med, lwr, upr]):
        raise ValueError("med, lwr, and upr must be non-negative.")

    def loss_lower(theta):
        tent_qs = lognorm.ppf(
            [(1 - conf_level) / 2, (1 + conf_level) / 2],
            s=theta[1],
            scale=np.exp(theta[0]),
        )
        if lwr == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = (
                abs(lwr - tent_qs[0]) / lwr + abs(upr - tent_qs[1]) / upr
            )
        return attained_loss

    def loss_median(theta):
        tent_qs = lognorm.ppf(
            [0.5, (1 + conf_level) / 2], s=theta[1], scale=np.exp(theta[0])
        )
        if med == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = (
                abs(med - tent_qs[0]) / med + abs(upr - tent_qs[1]) / upr
            )
        return attained_loss

    if med == 0:
        mustar = np.log(0.1)
    else:
        mustar = np.log(med)

    if fn_loss == "median":
        result = minimize(
            loss_median,
            x0=[mustar, 0.5],
            bounds=[(-5 * abs(mustar), 5 * abs(mustar)), (0, 15)],
            method="Nelder-mead",
            options={
                "xatol": 1e-6,
                "fatol": 1e-6,
                "maxiter": 1000,
                "maxfev": 1000,
            },
        )
    if fn_loss == "lower":
        result = minimize(
            loss_lower,
            x0=[mustar, 0.5],
            bounds=[(-5 * abs(mustar), 5 * abs(mustar)), (0, 15)],
            method="Nelder-mead",
            options={
                "xatol": 1e-8,
                "fatol": 1e-8,
                "maxiter": 5000,
                "maxfev": 5000,
            },
        )

    meanlog_opt, log_sdlog_opt = result.x
    
    return [meanlog_opt, log_sdlog_opt]


def get_lognormal_pars(med, lwr, upr, alpha = 0.95):
    def loss(theta):
        tent_qs = lognorm.ppf([(1 - alpha) / 2, (1 + alpha) / 2],
                              scale = exp(theta[0]),  # scale = exp(meanlog)
                              s = exp(theta[1]))  # s = sdlog
        if (lwr == 0):
            return abs(upr - tent_qs[1]) / upr
        else:
            return abs(lwr - tent_qs[0]) / lwr + abs(upr - tent_qs[1]) / upr

    mustar = log(med)
    bounds = [(-5 * abs(mustar), 5 * abs(mustar)), (None, log(10))]

    Opt = minimize(loss,
                   x0 = [mustar, 1/2],
                   method='L-BFGS-B',
                   bounds=bounds)
    meanlog_opt, log_sdlog_opt = Opt.x
    return [meanlog_opt, exp(log_sdlog_opt)]

def kl_lognormal(mu1, sigma1, mu2, sigma2):
    term1 = log(sigma2 / sigma1)
    term2 = (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
    kl = term1 + term2 - 0.5
    return kl

### Direct estimation from the ECDF
def fit_ln_CDF(x, Fhat, weighting = 1):
    K = len(x)
    if len(Fhat) != K:
        raise ValueError("Size mismatch between x and Fhat.")

    log_probs = log(Fhat)

    if weighting == 1:
        weights = 1 / (Fhat * (1 - Fhat)) * 1 / np.abs(log_probs)
    else:
        weights = np.ones(K)

    def opt_cdf_diff(par, ws = weights):
        mu = par[0]
        sigma = np.exp(par[1])
        theo_logF = lognorm.logcdf(x,
                                   s = sigma,
                                   scale = np.exp(mu))
        loss = np.sum(ws * np.abs(log_probs - theo_logF))
        return loss

    mustar = np.mean(np.log(x))
    Opt = minimize(opt_cdf_diff,
                   x0 = np.array([mustar, 1/2]),
                   bounds = [(-5 * abs(mustar), 5 * abs(mustar)), (None, log(10))], 
                   method="Nelder-mead",
                    options={
                        "xatol": 1e-6,
                        "fatol": 1e-6,
                        "maxiter": 1000,
                        "maxfev": 1000}
)

    return [Opt.x[0], np.exp(Opt.x[1])]

### Find log-normal which minimises the sum of KLs
def minimize_opt_fn1(method, x0, J, ln_approx):
    def opt_fn1(par):
        kls = np.full(J, np.nan)
        for j in range(0, J):
            kls[j] = kl_lognormal(mu1 = par[0],
                                  sigma1 = np.exp(par[1]),
                                  mu2 = ln_approx['mu'][j],
                                  sigma2 = ln_approx['sigma'][j])

        return np.sum(kls)

    result_opt_fn1 = minimize(opt_fn1,
                              x0=x0,
                              method=method)
    return result_opt_fn1

### Find the alpha which minimises the sum of KLs (Log-pooling)
def get_lognormal_pool_pars(ms, vs, weights):
    pars = pyLogPool.pool_par_gauss(alpha = weights, m = ms, v = vs)
    return pars

def minimize_opt_fn2(x0, J, ln_approx):

    def opt_fn2(par):
        kls = np.full(J, np.nan)
        ws = pyLogPool.alpha_01(par)
        pool = get_lognormal_pool_pars(ms = ln_approx['mu'],
                                       vs = ln_approx['sigma']**2,
                                       weights = ws)
        for j in range(0,J):
            kls[j] = kl_lognormal(mu1 = pool[0],
                                  sigma1 = pool[1],
                                  mu2 = ln_approx['mu'][j],
                                  sigma2 = ln_approx['sigma'][j])

        return np.sum(kls)

    result_opt_fn2 = minimize(opt_fn2,
                              x0=x0)

    return result_opt_fn2

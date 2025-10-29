import numpy as np
import scipy.stats as st

def estimate_rho_correlation(history):
    ranks = st.rankdata(history) / (len(history) + 1)
    u = ranks
    z = st.norm.ppf(u)  # Gaussian scores

    # lagged pairs
    z_lag, z_curr = z[:-1], z[1:]
    rho = np.corrcoef(z_lag, z_curr)[0, 1]
    return rho

def sample_path(forecast_marginals, rho, random_state=None):
    horizon = len(forecast_marginals)
    rng = np.random.default_rng(random_state)
    path = np.zeros(horizon)
    x_0 = forecast_marginals[0].rvs(size=1, random_state=rng)[0]
    path[0] = x_0
    for j in range(1, horizon):
        u_prev = forecast_marginals[j-1].cdf(path[j-1])
        z_prev = st.norm.ppf(u_prev)
        z = rng.normal(loc=rho * z_prev, scale=np.sqrt(1 - rho**2))
        u = st.norm.cdf(z)
        x = forecast_marginals[j].ppf(u)
        path[j] = x
    return path

def cumulative_estimation(forecast_marginals, rho, n_paths=1000):
    paths = [sample_path(forecast_marginals, rho) for _ in range(n_paths)]
    samples = [sum(p) for p in paths]
    return list(np.percentile(samples, [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]))




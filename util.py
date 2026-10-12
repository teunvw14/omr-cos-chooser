import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def bs_price_call(K, S_0, r, sigma, tau, rng):
    d1 = (np.log(S_0/K) + (r + 0.5*sigma*sigma)*tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S_0 * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

def bs_price_put(K, S_0, r, sigma, tau):
    d1 = (np.log(S_0/K) + (r + 0.5*sigma*sigma)*tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S_0 * norm.cdf(-d1)

def mc_price(payoff, paths, r, tau):
    final_values = paths[:, -1]
    payoffs = payoff(final_values)
    discount = np.exp(-r * tau)
    return discount * payoffs.mean()

def gbm_generate_paths(S_0, mu, sigma, tau, n_paths, n_steps, rng):
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = np.log(S_0)

    dt = tau / n_steps

    for i in range(n_steps):
        X[:, i+1] = X[:, i] + (mu - 0.5*sigma*sigma) * dt + sigma * np.sqrt(dt) * rng.normal(size=n_paths)

    return np.exp(X)

def merton_generate_paths(S_0, mu, sigma, jump_rate, jump_mean, jump_std, tau, n_paths, n_steps, rng):
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = np.log(S_0)

    dt = tau / n_steps
    for i in range(n_steps):
        jumps = rng.poisson(dt * jump_rate, size=n_paths)
        X[:, i+1] = X[:, i] + mu * dt + sigma * np.sqrt(dt) * rng.normal(size=n_paths) + jumps * (jump_mean + jump_std * rng.normal(size=n_paths))

    return np.exp(X)

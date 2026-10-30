import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def bs_price_call(K, S_0, r, sigma, tau):
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

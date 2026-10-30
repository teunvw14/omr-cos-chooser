import numpy as np
from scipy.stats import norm

def bs_call(S0, K, r, tau, sigma, q=0):
    d1 = (np.log(S0 / K)
            + (r - q + 0.5 * sigma **2) * tau
            ) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    return (S0 * np.exp(-q*tau) * norm.cdf(d1)
            - K * np.exp(-r*tau) * norm.cdf(d2))
    
def bs_put(S0, K, r, tau, sigma, q=0):
    d1 = (np.log(S0 / K)
            + (r - q + 0.5 * sigma **2) * tau
            ) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    return (K * np.exp(-r*tau) * norm.cdf(-d2)
            - S0 * np.exp(-q*tau) * norm.cdf(-d1))

def bs_simple_chooser(S0, K, r, tau, tc, sigma, q=0.0):
    call = bs_call(S0, K, r, tau, sigma, q)
    put  = bs_put(S0, K * np.exp(-r * (tau - tc)), r, tc, sigma, q) # With adjusted strike and maturity tc

    return call + put
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm, multivariate_normal

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

def get_equality_point_gbm(K_C, K_P, tau_C, tau_P, S0, r, sigma):
    call_value = lambda x: bs_price_call(K_C, x, r, sigma, tau_C)
    put_value = lambda x: bs_price_put(K_P, x, r, sigma, tau_P)
    g = lambda x: call_value(x) - put_value(x)

    a = 0.00001
    if g(S0) > 0:
        b = S0
        return optimize.brentq(g, a, b)
    else:
        b = S0
        while g(b) < 0:
            b *= 4
        return optimize.brentq(g, a, b)


# 2-dimensional normal distribution CDF
def N2_cdf(x, y, rho):
    cov = [
        [1, rho],
        [rho, 1]
    ]
    return multivariate_normal([0, 0], cov).cdf([x, y])

def price_complex_chooser_gbm_analytic(S0, r, sigma, K_C, K_P, T_C, T_P, tc):
    tau_C = T_C - tc
    tau_P = T_P - tc
    rho_C = np.sqrt(tc/T_C)
    rho_P = np.sqrt(tc/T_P)

    d_plus = lambda tau, x, K: (np.log(x / K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d_minus = lambda tau, x, K: (np.log(x / K) + (r - 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    z = get_equality_point_gbm(K_C, K_P, tau_C, tau_P, S0, r, sigma)

    tau_1 = tc
    tau_2_C = T_C
    tau_2_P = T_P
    return S0 * N2_cdf(
        d_plus(tau_1, S0, z), d_plus(tau_2_C, S0, K_C), rho_C
    ) - K_C * np.exp(-r * tau_2_C) * N2_cdf(
        d_minus(tau_1, S0, z), d_minus(tau_2_C, S0, K_C), rho_C
    ) + K_P * np.exp(-r * tau_2_P) * N2_cdf(
        -d_minus(tau_1, S0, z), -d_minus(tau_2_P, S0, K_P), rho_P
    ) - S0 * N2_cdf(
         -d_plus(tau_1, S0, z), -d_plus(tau_2_P, S0, K_P), rho_P
    )
import numpy as np
from math import pi
from scipy import optimize
from numpy import exp, cos, sin, log, sqrt
from util import bs_price_call, bs_price_put, price_complex_chooser_gbm_analytic, N2_cdf
from cos_chooser import price_complex_chooser_cos, bs_cf
import matplotlib.pyplot as plt

def graph_maturity_vs_price_tc():
    # K_C = 105.0
    # K_P = 95.0
    # T_C = 1.0
    T_dists = np.linspace(0, 9, num = 10)
    T_P = 1.0
    S_0 = 100.0
    X_0 = log(S_0)
    K = 100
    K_C = K
    K_P = K
    r = 0.02
    sigma = 0.2
    N = 128
    
    tcs = np.linspace(0.1, 0.9, num=5)

    # K_dists = np.arange(-100, 100, step=10)
    prices = {}
    for tc in tcs:
        prices.update({tc: []})

        for T_dist in T_dists:
            T_C = T_P + T_dist
            T_max = max(T_C, T_P)

            a = X_0 - 10 * sigma * (T_max**0.5)
            b = X_0 + 10 * sigma * (T_max**0.5)
            cf_C = bs_cf(r, sigma, T_C - tc)
            cf_P = bs_cf(r, sigma, T_P - tc)
            cf_out = bs_cf(r, sigma, tc)
            # K_C = K - K_dist/2
            # K_P = K + K_dist/2
            a_C = log(K_C) - 10 * sigma * (T_max**0.5)
            b_C = log(K_C) + 10 * sigma * (T_max**0.5)
            a_P = log(K_P) - 10 * sigma * (T_max**0.5)
            b_P = log(K_P) + 10 * sigma * (T_max**0.5)
            price = price_complex_chooser_cos(cf_out, cf_C, cf_P, S_0, r, K_C, K_P, T_C, T_P, tc, N, a, b, a_C, b_C, a_P, b_P)
            prices[tc].append(price)
        label = r"$T_1=" + f"{tc:.2f}$"
        plt.plot(T_dists + 1, prices[tc], label=label)

    plt.xlabel("Call Maturity $T_C$")
    plt.ylabel("Chooser Option Price")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.show()


def main():
    graph_maturity_vs_price_tc()


if __name__ == "__main__":
    main()
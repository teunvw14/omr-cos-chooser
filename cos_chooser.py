import numpy as np
from math import pi
from scipy import integrate
from numpy import exp, cos, sin, log
from util import bs_price_call, bs_price_put
import matplotlib.pyplot as plt

# Calculate \chi_k
def get_chi_k(k, c, d, a_in, b_in):
    c = max(c, a_in)
    omega_k = k * pi / (b_in - a_in)
    return (1/ (1 + omega_k**2)) \
        * (
            exp(d) * (cos(omega_k*(d - a_in)) + omega_k * sin(omega_k*(d - a_in)))
            -
            exp(c) * (cos(omega_k*(c - a_in)) + omega_k * sin(omega_k*(c - a_in)))
        )


# Calculate \phi_k
def get_phi_k(k, c, d, a_in, b_in):
    c = max(c, a_in)
    if k == 0:
        return d - c
    omega_k = k * pi / (b_in - a_in)
    return (sin(omega_k * (d - a_in)) - sin(omega_k * (c - a_in))) / omega_k


def get_V(K, a_in, b_in, N):
    # safeguard for if a b_in is chosen that is much too small
    if log(K) >= b_in:
        return np.zeros(N)
    
    result = []
    for k in range(N):
        chi_k = get_chi_k(k, log(K), b_in, a_in, b_in)
        phi_k = get_phi_k(k, log(K), b_in, a_in, b_in)
        V_k = (2 / (b_in - a_in)) * (chi_k - K * phi_k)
        result.append(V_k)
    return np.array(result)


def get_Ic(k, n, a, b, a_in, b_in):
    omega_k = k * pi / (b_in - a_in)
    nu_n = n * pi / (b - a)
    if k == 0 and n == 0:
        return 2
    if np.isclose(omega_k, nu_n):
        f = lambda x: sin(2*omega_k*x - omega_k*(a + a_in)) / (2*omega_k)
        return cos(omega_k * (a - a_in)) + (1/(b-a)) * (f(b) - f(a))
    else:
        f = lambda x: (sin((omega_k - nu_n)*x - (omega_k*a_in - nu_n*a))/ (omega_k - nu_n)) + (sin((omega_k + nu_n)*x - (omega_k*a_in + nu_n*a))/ (omega_k + nu_n))
        return (1/(b-a)) * (f(b) - f(a))


def get_Is(k, n, a, b, a_in, b_in):
    omega_k = k * pi / (b_in - a_in)
    nu_n = n * pi / (b - a)
    if k == 0 and n == 0:
        return 0
    if np.isclose(omega_k, nu_n):
        f = lambda x: cos(2*omega_k*x - omega_k*(a + a_in)) / (2*omega_k)
        return sin(omega_k * (a - a_in)) - (1/(b-a)) * (f(b) - f(a))
    else:
        f = lambda x: (cos((omega_k + nu_n)*x - (omega_k*a_in + nu_n*a))/ (omega_k + nu_n)) + (cos((omega_k - nu_n)*x - (omega_k*a_in - nu_n*a))/ (omega_k - nu_n))
        return -(1/(b-a)) * (f(b) - f(a))


def chooser_I3(n, a, b, a_in, b_in, N, A, B):
    Ic = np.array([get_Ic(k, n, a, b, a_in, b_in) for k in range(N)])
    Is = np.array([get_Is(k, n, a, b, a_in, b_in) for k in range(N)])

    terms = A * Ic - B * Is
    terms[0] *= 0.5
    I3 = np.sum(terms)
    return I3


# Compute H_n
def chooser_outer_cosine_coeff(n, K, r, T, tc, N, a, b, a_in, b_in, A, B):
    x_1 = np.log(K) - r * (T - tc)
    nu_n = n * np.pi / (b-a)

    if n == 0:            
        I_1 = 2 * K * exp(-r * (T-tc)) * (x_1-a) / (b-a)
        I_2 = 2 * (exp(x_1) - exp(a)) / (b-a)
        I_3 = chooser_I3(0, a, b, a_in, b_in, N, A, B)
    else:
        I_1 = 2 * K * exp(-r * (T-tc)) * sin(nu_n * (x_1-a)) / ((b-a) * nu_n)
        I_2 = 2 * (exp(x_1) * ((cos(nu_n * (x_1-a)) + nu_n * sin(nu_n * (x_1-a)))) - exp(a)) / ((b-a) * (nu_n**2 + 1))        
        I_3 = chooser_I3(n, a, b, a_in, b_in, N, A, B)

    H_out = I_1 - I_2 + I_3

    return H_out


def price_chooser_cos(cf_out, cf_in, S0, K, r, T, tc, N, a, b, a_in, b_in):
    x_0 = np.log(S0)

    k = np.arange(N)
    omega = k * pi / (b_in - a_in)
    psi = cf_in(omega)

    V = get_V(K, a_in, b_in, N)
    A = exp(-r * (T-tc)) * psi.real * V
    B = exp(-r * (T-tc)) * psi.imag * V

    terms = np.zeros(N)
    for n in range(N):
        nu_n = n * np.pi / (b-a)
        H_n = chooser_outer_cosine_coeff(n, K, r, T, tc, N, a, b, a_in, b_in, A, B)
        terms[n] = np.real(cf_out(nu_n) * exp(1j * nu_n * (x_0 - a))) * H_n

    terms[0] *= 0.5

    discount = exp(-r * tc)
    return discount * np.sum(terms) 


# Characteristic function of X(t) - X(s) under Black-Scholes
def bs_cf(r, sigma, tau):
    return lambda u: np.exp(1j * u * (r - 0.5 * sigma**2) * tau - 0.5 * sigma**2 * u**2 * tau)


def main():
    K = 100.0
    S_0 = 100.0
    X_0 = log(S_0)
    r = 0.02
    sigma = 0.2
    T = 1.0
    tc = 0.5

    # cos method parameters
    a_in = X_0 - 10 * sigma * (T**0.5)
    b_in = X_0 + 10 * sigma * (T**0.5)
    a = X_0 - 10 * sigma * (T**0.5)
    b = X_0 + 10 * sigma * (T**0.5)
    N = np.logspace(2, 8, base=2, num=10, dtype=int)
    cf_out = bs_cf(r, sigma, tc)
    cf_in = bs_cf(r, sigma, T - tc)

    # Calculate the COS price for an increasing number of terms, show in a plot.
    price_analytic = bs_price_call(K, S_0, r, sigma, T) + bs_price_put(K * exp(-r * (T-tc)), S_0, r, sigma, tc)
    prices_cos = np.array([
        price_chooser_cos(cf_out, cf_in, S_0, K, r, T, tc, NN, a, b, a_in, b_in)
        for NN in N
    ])
    errs = np.abs(prices_cos - price_analytic)

    print(errs)

    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("Error (Log scale)")
    plt.title("Error for COS Pricing the Chooser Option")
    plt.plot(N, errs)
    plt.show()

if __name__ == "__main__":
    main()

import numpy as np
from math import pi
from scipy import optimize
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


def get_V_call(K, a_in, b_in, N):
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

def get_V_put(K, a_in, b_in, N):
    # payoff is zero on [a_in, b_in] if log(K) <= a_in
    if log(K) <= a_in:
        return np.zeros(N)

    d = min(log(K), b_in)
    result = []
    for k in range(N):
        chi_k = get_chi_k(k, a_in, d, a_in, b_in)
        phi_k = get_phi_k(k, a_in, d, a_in, b_in)
        V_k = (2 / (b_in - a_in)) * (-chi_k + K * phi_k)
        result.append(V_k)
    return np.array(result)


def get_Ic(k, n, a, b, a_in, b_in, c, d):
    omega_k = k * pi / (b_in - a_in)
    nu_n = n * pi / (b - a)
    if k == 0 and n == 0:
        return 2 * (d-c)/(b-a)
    if np.isclose(omega_k, nu_n):
        f = lambda x: sin(2*omega_k*x - omega_k*(a + a_in)) / (2*omega_k)
        return ((d-c)/(b-a)) * cos(omega_k * (a - a_in)) + (f(d) - f(c))/(b-a)
    else:
        f = lambda x: (sin((omega_k - nu_n)*x - (omega_k*a_in - nu_n*a))/ (omega_k - nu_n)) + (sin((omega_k + nu_n)*x - (omega_k*a_in + nu_n*a))/ (omega_k + nu_n))
        return (f(d) - f(c))/(b-a)


def get_Is(k, n, a, b, a_in, b_in, c, d):
    omega_k = k * pi / (b_in - a_in)
    nu_n = n * pi / (b - a)
    if k == 0 and n == 0:
        return 0
    if np.isclose(omega_k, nu_n):
        f = lambda x: cos(2*omega_k*x - omega_k*(a + a_in)) / (2*omega_k)
        return ((d-c)/(b-a)) * sin(omega_k * (a - a_in)) - (f(d) - f(c)) / (b-a)
    else:
        f = lambda x: (cos((omega_k + nu_n)*x - (omega_k*a_in + nu_n*a))/ (omega_k + nu_n)) + (cos((omega_k - nu_n)*x - (omega_k*a_in - nu_n*a))/ (omega_k - nu_n))
        return -(f(d) - f(c)) / (b-a)


def chooser_I3(n, a, b, a_in, b_in, N, A, B, c, d):
    Ic = np.array([get_Ic(k, n, a, b, a_in, b_in, c, d) for k in range(N)])
    Is = np.array([get_Is(k, n, a, b, a_in, b_in, c, d) for k in range(N)])

    terms = A * Ic - B * Is
    terms[0] *= 0.5
    I3 = np.sum(terms)
    return I3


# Compute H_n for a simple chooser option
def simple_chooser_outer_cosine_coeff(n, K, r, T, tc, N, a, b, a_in, b_in, A, B):
    x_1 = np.log(K) - r * (T - tc)
    nu_n = n * np.pi / (b-a)

    if n == 0:            
        I_1 = 2 * K * exp(-r * (T-tc)) * (x_1-a) / (b-a)
        I_2 = 2 * (exp(x_1) - exp(a)) / (b-a)
        I_3 = chooser_I3(0, a, b, a_in, b_in, N, A, B, a, b)
    else:
        I_1 = 2 * K * exp(-r * (T-tc)) * sin(nu_n * (x_1-a)) / ((b-a) * nu_n)
        I_2 = 2 * (exp(x_1) * ((cos(nu_n * (x_1-a)) + nu_n * sin(nu_n * (x_1-a)))) - exp(a)) / ((b-a) * (nu_n**2 + 1))        
        I_3 = chooser_I3(n, a, b, a_in, b_in, N, A, B, a, b)

    H_out = I_1 - I_2 + I_3

    return H_out



def get_AB(cf, r, tau, V, a_in, b_in, N):
    k = np.arange(N)
    omega = k * pi / (b_in - a_in)
    psi = cf(omega)

    discount = exp(-r * tau)
    A = discount * psi.real * V
    B = discount * psi.imag * V

    return A, B


# Compute H_n for a complex chooser option
def complex_chooser_outer_cosine_coeff(n, N, a, b, a_C, b_C, a_P, b_P, A_C, B_C, A_P, B_P, x1):

    H_C = chooser_I3(n, a, b, a_C, b_C, N, A_C, B_C, x1, b)
    H_P = chooser_I3(n, a, b, a_P, b_P, N, A_P, B_P, a, x1)

    H_out = H_C + H_P

    return H_out


def price_complex_chooser_cos(cf_out, cf_C, cf_P, S0, r, K_C, K_P, T_C, T_P, tc, N, a, b, a_C, b_C, a_P, b_P):
    x_0 = np.log(S0)
    tau_C = T_C - tc
    tau_P = T_P - tc

    V_C = get_V_call(K_C, a_C, b_C, N)
    V_P = get_V_put(K_P, a_P, b_P, N)
    A_C, B_C = get_AB(cf_C, r, tau_C, V_C, a_C, b_C, N)
    A_P, B_P = get_AB(cf_P, r, tau_P, V_P, a_P, b_P, N)
    x1 = get_equality_point(a, b, cf_C, tau_C, V_C, a_C, b_C, cf_P, tau_P, V_P, a_P, b_P, r)

    terms = np.zeros(N)
    for n in range(N):
        nu_n = n * np.pi / (b-a)
        H_n = complex_chooser_outer_cosine_coeff(n, N, a, b, a_C, b_C, a_P, b_P, A_C, B_C, A_P, B_P, x1)
        terms[n] = np.real(cf_out(nu_n) * exp(1j * nu_n * (x_0 - a))) * H_n

    terms[0] *= 0.5

    discount = exp(-r * tc)
    return discount * np.sum(terms) 

def price_simple_chooser_cos(cf_out, cf_in, S0, r, K, T, tc, N, a, b, a_in, b_in):
    x_0 = np.log(S0)
    V_call = get_V_call(K, a_in, b_in, N)
    A, B = get_AB(cf_in, r, T-tc, V_call, a_in, b_in, N)

    terms = np.zeros(N)
    for n in range(N):
        nu_n = n * np.pi / (b-a)
        H_n = simple_chooser_outer_cosine_coeff(n, K, r, T, tc, N, a, b, a_in, b_in, A, B)
        terms[n] = np.real(cf_out(nu_n) * exp(1j * nu_n * (x_0 - a))) * H_n

    terms[0] *= 0.5

    discount = exp(-r * tc)
    return discount * np.sum(terms) 


# Characteristic function of X(t) - X(s) under Black-Scholes
def bs_cf(r, sigma, tau):
    return lambda u: np.exp(1j * u * (r - 0.5 * sigma**2) * tau - 0.5 * sigma**2 * u**2 * tau)


# Option value at ln(S) = x, using payoff coefficients V and the characteristic
# function of the log of the asset
def cos_value(x, cf, r, tau, V, a_in, b_in):
    x = np.atleast_1d(x)
    N = len(V)
    omega = np.arange(N) * pi / (b_in - a_in)
    psi = cf(omega)                                    # shape (N,)
    M = np.exp(1j * np.outer(x - a_in, omega))         # shape (len(x), N)
    terms = (psi * M).real * V                         # shape (len(x), N)
    terms[:, 0] *= 0.5
    return exp(-r * tau) * terms.sum(axis=1)

# Find the point x^* = ln(S^*) where the value of the call and put are equal.
def get_equality_point(a, b, cf_C, tau_C, V_C, a_C, b_C, 
                        cf_P, tau_P, V_P, a_P, b_P, r):
    call_value = lambda x: cos_value(x, cf_C, r, tau_C, V_C, a_C, b_C)[0]
    put_value = lambda x: cos_value(x, cf_P, r, tau_P, V_P, a_P, b_P)[0]

    g = lambda x: call_value(x) - put_value(x)

    # a represents the lowest price, so it should have higher put value; if not
    # then we use a
    if g(a) >= 0:
        return a
    if g(b) <= 0:
        return b

    return optimize.brentq(g, a, b)


def plot_convergence_in_N():
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
    N = np.array([4, 8, 16, 32, 64, 128])
    # N = np.logspace(4, 7, base=2, num=6, dtype=int)
    cf_out = bs_cf(r, sigma, tc)
    cf_in = bs_cf(r, sigma, T - tc)

    # Calculate the COS price for an increasing number of terms, show in a plot.
    price_analytic = bs_price_call(K, S_0, r, sigma, T) + bs_price_put(K * exp(-r * (T-tc)), S_0, r, sigma, tc)
    prices_cos = np.array([
        price_simple_chooser_cos(cf_out, cf_in, S_0, r, K, T, tc, NN, a, b, a_in, b_in)
        for NN in N
    ])
    errs = np.abs(prices_cos - price_analytic)

    print(f"N: {N}")
    print(f"values: {prices_cos}")
    print(f"errs: {errs}")

    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("Error (Log scale)")
    plt.plot(N, errs)
    plt.grid(alpha=0.5)
    
    plt.show()


def test_equality_point():
    K_C = 120.0
    K_P = 95.0
    T_C = 2.0
    T_P = 3.0
    S_0 = 100.0
    X_0 = log(S_0)
    r = 0.02
    sigma = 0.2
    T_max = max(T_C, T_P)
    tc = 0.5

    # cos method parameters
    a_in = X_0 - 10 * sigma * (T_max**0.5)
    b_in = X_0 + 10 * sigma * (T_max**0.5)
    a = X_0 - 10 * sigma * (T_max**0.5)
    b = X_0 + 10 * sigma * (T_max**0.5)
    N = 64

    cf_C = bs_cf(r, sigma, T_C - tc)
    cf_P = bs_cf(r, sigma, T_P - tc)
    tau_C = T_C - tc
    tau_P = T_P - tc
    V_c = get_V_call(K_C, a_in, b_in, N)
    V_p = get_V_put(K_P, a_in, b_in, N)
    x_star = get_equality_point(a, b, cf_C, tau_C, V_c, a, b, cf_P, tau_P, V_p, a, b, r)
    S_star = np.exp(x_star)
    print(S_star)

    analytic = optimize.brentq(lambda x: bs_price_call(K_C, exp(x), r, sigma, T_C - tc)
                - bs_price_put(K_P, exp(x), r, sigma, T_P - tc), a, b)
    print(np.exp(analytic))

    return

def test_complex_chooser_cos_pricer_1():
    K = 100
    T = 1
    S_0 = 100.0
    X_0 = log(S_0)
    r = 0.02
    sigma = 0.2
    tc = 0.5

    # cos method parameters
    a_in = X_0 - 10 * sigma * (T**0.5)
    b_in = X_0 + 10 * sigma * (T**0.5)
    a = X_0 - 10 * sigma * (T**0.5)
    b = X_0 + 10 * sigma * (T**0.5)
    N = 128

    cf = bs_cf(r, sigma, T - tc)
    cf_out = bs_cf(r, sigma, tc)

    price_clx = price_complex_chooser_cos(cf_out, cf, cf, S_0, r, K, K, T, T, tc, N, a, b, a, b, a, b)
    price_smp = price_simple_chooser_cos(cf_out, cf, S_0, r, K, T, tc, N, a, b, a_in, b_in)
    print(price_clx)
    print(price_smp)

def test_complex_chooser_cos_pricer_2():
    K_C = 105.0
    K_P = 95.0
    T_C = 1.0
    T_P = 0.8
    S_0 = 100.0
    X_0 = log(S_0)
    r = 0.02
    sigma = 0.2
    T_max = max(T_C, T_P)
    tc = 0.5

    # cos method parameters
    a_in = X_0 - 10 * sigma * (T_max**0.5)
    b_in = X_0 + 10 * sigma * (T_max**0.5)
    a = X_0 - 10 * sigma * (T_max**0.5)
    b = X_0 + 10 * sigma * (T_max**0.5)
    N = 64

    cf_C = bs_cf(r, sigma, T_C - tc)
    cf_P = bs_cf(r, sigma, T_P - tc)
    cf_out = bs_cf(r, sigma, tc)

    price = price_complex_chooser_cos(cf_out, cf_C, cf_P, S_0, r, K_C, K_P, T_C, T_P, tc, N, a, b, a, b, a, b)
    print(price)


def main():
    # plot_convergence_in_N()
    test_equality_point()
    # test_complex_chooser_cos_pricer_1()



if __name__ == "__main__":
    main()

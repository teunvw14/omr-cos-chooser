import numpy as np
from scipy.optimize import brentq
from integrals import build_Ic, build_Is

def chi_psi(a, b, c, d, k):
    omega = k * np.pi / (b - a)

    # Psi coefficients
    psi = np.empty_like(omega)
    psi[0] = d - c
    psi[1:] = (
        np.sin(omega[1:] * (d - a))
        - np.sin(omega[1:] * (c - a))
    ) / omega[1:]

    # Chi coefficients
    chi = (
        np.cos(omega * (d - a)) * np.exp(d)
        - np.cos(omega * (c - a)) * np.exp(c)
        + omega * np.sin(omega * (d - a)) * np.exp(d)
        - omega * np.sin(omega * (c - a)) * np.exp(c)
    ) / (1.0 + omega**2)

    return chi, psi

def truncation_interval(c1, c2, c4, L):
    width = L * np.sqrt(c2 + np.sqrt(c4))
    a = c1 - width
    b = c1 + width

    return a, b

def put_coeffs(a, b, k):
    c = a
    d = 0.0
    chi, psi = chi_psi(a, b, c, d, k)
    H_k = 2.0 / (b - a) * (psi - chi)

    return H_k

def call_coeffs(a, b, k):
    c = 0.0
    d = b

    chi, psi = chi_psi(a, b, c, d, k,)
    H_k = (2.0 / (b - a) * (chi - psi)    )

    return H_k

def cos_value(phi, x, r, tau, a, b, Hk):
    k = np.arange(len(Hk))
    u = k * np.pi / (b - a)

    temp = phi(u, tau) * Hk
    temp[0] *= 0.5

    return (
        np.exp(-r * tau)
        * np.real(
            np.sum(
                np.exp(1j * u * (x - a)) * temp
            )
        )
    )

def cos_put_price(phi, cumulants, S0, K, r, tau, N, L):
    x0 = np.log(S0 / K)

    c1, c2, c4 = cumulants(tau)

    a, b = truncation_interval(x0+c1, c2, c4, L)

    k = np.arange(N)

    Hk = put_coeffs(a, b, k)

    put = K * cos_value(phi=phi, x=x0, r=r, tau=tau, a=a, b=b, Hk=Hk)
    
    return put


def cos_call_price(phi, cumulants, S0, K, r, tau, N, L):
    put = cos_put_price(phi=phi, cumulants=cumulants, S0=S0, K=K, r=r, tau=tau, N=N, L=L)

    call = put + S0 - K * np.exp(-r * tau)

    return call

def chooser_boundary(
    phi,
    cumulants,
    r,
    tau,
    N,
    L,
    Kc,
    Kp,
    xmin=None,
    xmax=None,
):

    if xmin is None:
        xmin = np.log(min(Kc, Kp)) - 5.0

    if xmax is None:
        xmax = np.log(max(Kc, Kp)) + 5.0

    def f(x):
        S = np.exp(x)

        call_value = cos_call_price(
            phi=phi,
            cumulants=cumulants,
            S0=S,
            K=Kc,
            r=r,
            tau=tau,
            N=N,
            L=L,
        )

        put_value = cos_put_price(
            phi=phi,
            cumulants=cumulants,
            S0=S,
            K=Kp,
            r=r,
            tau=tau,
            N=N,
            L=L,
        )

        return call_value - put_value

    f_left = f(xmin)
    f_right = f(xmax)

    if f_left * f_right > 0.0:
        raise ValueError(
            "Could not bracket the chooser boundary: "
            f"f(xmin)={f_left:.6e}, "
            f"f(xmax)={f_right:.6e}."
        )

    return brentq(
        f,
        xmin,
        xmax,
        xtol=1e-12,
        rtol=1e-12,
    )



def chooser_coeffs(
    A_call,
    B_call,
    A_put,
    B_put,
    omega,
    nu,
    a,
    aprime,
    b,
    x_star,
    Kc,
    Kp,
):
    # Put region: [a, x_star]
    Ic_put = build_Ic(
        omega, nu, a, aprime, b, a, x_star,
    )

    Is_put = build_Is(
        omega, nu, a, aprime, b, a, x_star,
    )

    H_put = Kp * (
      A_put @ Ic_put
        -B_put @ Is_put
    )

    # Call region: [x_star, b]
    Ic_call = build_Ic(
        omega, nu, a, aprime, b, x_star, b,
    )

    Is_call = build_Is(
        omega, nu, a, aprime, b, x_star, b,
    )

    H_call = Kc * (
        A_call @ Ic_call
        - B_call @ Is_call
    )

    return H_put + H_call


def cos_chooser_price(
    phi,
    cumulants,
    S0,
    Kc,
    Kp,
    r,
    tc,
    T,
    N=256,
    L=10,
    M=5000,
):
    """
    Numerical COS chooser pricer.

    Choice at tc between:
        Call(Kc, T)
        Put(Kp, T)

    Uses numerical midpoint integration for the
    chooser payoff coefficients.
    """

    tau = T - tc

    # --------------------------------------------------
    # 1. Truncation interval at choice time
    # --------------------------------------------------

    x0 = np.log(S0 / Kc)

    c1, c2, c4 = cumulants(tc)

    a, b = truncation_interval(
        x0 + c1, c2, c4, L
    )

    # --------------------------------------------------
    # 2. Numerical integration grid
    # --------------------------------------------------

    dx = (b - a) / M

    x = a + (np.arange(M) + 0.5) * dx

    S = Kc * np.exp(x)

    # --------------------------------------------------
    # 3. Continuation call and put values
    # --------------------------------------------------

    payoff = np.empty(M)

    for j in range(M):

        call = cos_call_price(
            phi=phi,
            cumulants=cumulants,
            S0=S[j],
            K=Kc,
            r=r,
            tau=tau,
            N=N,
            L=L,
        )

        put = cos_put_price(
            phi=phi,
            cumulants=cumulants,
            S0=S[j],
            K=Kp,
            r=r,
            tau=tau,
            N=N,
            L=L,
        )

        payoff[j] = max(call, put)

    # --------------------------------------------------
    # 4. Numerical COS payoff coefficients
    # --------------------------------------------------

    k = np.arange(N)

    u = k * np.pi / (b - a)

    cos_matrix = np.cos(
        np.outer(u, x - a)
    )

    Hk = (
        2.0 / (b - a)
        * dx
        * (cos_matrix @ payoff)
    )

    # --------------------------------------------------
    # 5. COS expectation
    # --------------------------------------------------

    phi_values = phi(u, tc)

    weights = np.ones(N)
    weights[0] = 0.5

    price = np.exp(-r * tc) * np.sum(
        weights
        * Hk
        * np.real(
            phi_values * np.exp(1j * u * (x0 - a))
        )
    )

    return price
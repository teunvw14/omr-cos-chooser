import numpy as np
from scipy.optimize import brentq
from integrals import build_Ic, build_Is
from coefficients import inner_coeffs, continuation_values

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

def put_coeffs(K,a, b, k): # Strike is passed to allow for Kc different from Kp 
    c = a
    d = np.clip(np.log(K), a, b)
    chi, psi = chi_psi(a, b, c, d, k)
    H_k = 2.0 / (b - a) * (K*psi - chi)

    return H_k

def call_coeffs(K,a, b, k):
    c = np.clip(np.log(K), a, b)
    d = b

    chi, psi = chi_psi(a, b, c, d, k,)
    H_k = (2.0 / (b - a) * (chi - K*psi)    )

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
    x0 = np.log(S0)

    c1, c2, c4 = cumulants(tau)

    a, b = truncation_interval(x0+c1, c2, c4, L)

    k = np.arange(N)

    Hk = put_coeffs(K,a, b, k)

    put = cos_value(phi=phi, x=x0, r=r, tau=tau, a=a, b=b, Hk=Hk)
    
    return put


def cos_call_price(phi, cumulants, S0, K, r, tau, N, L):
    put = cos_put_price(
        phi=phi,
        cumulants=cumulants,
        S0=S0,
        K=K, 
        r=r, 
        tau=tau, 
        N=N, 
        L=L)

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

    return brentq(f, xmin, xmax, xtol=1e-12, rtol=1e-12)



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
):
    # Put region: [a, x_star]
    Ic_put = build_Ic(
        omega, nu, a, aprime, b, a, x_star,
    )

    Is_put = build_Is(
        omega, nu, a, aprime, b, a, x_star,
    )

    H_put = (A_put @ Ic_put - B_put @ Is_put)

    # Call region: [x_star, b]
    Ic_call = build_Ic(
        omega, nu, a, aprime, b, x_star, b,
    )

    Is_call = build_Is(
        omega, nu, a, aprime, b, x_star, b,
    )

    H_call = (A_call @ Ic_call - B_call @ Is_call)

    return H_put + H_call

def cos_chooser_analytic(phi, cumulants, S0, Kc, Kp, r, tc, tau, Nin, Nout, L=10):
    x0, tau = np.log(S0), tau - tc

    # intervals
    c1, c2, c4 = cumulants(tc)
    a, b = truncation_interval(x0 + c1, c2, c4, L)
    e1, e2, e4 = cumulants(tau)
    d = L*np.sqrt(e2 + np.sqrt(e4))
    ap, bp = a + e1 - d, b + e1 + d

    # inner expansions: same interval, so same omega for call and put
    k  = np.arange(Nin)
    Hc = call_coeffs(Kc, ap, bp, k); Hc[0] *= 0.5
    Hp = put_coeffs(Kp, ap, bp, k); Hp[0] *= 0.5
    Ac, Bc = inner_coeffs(phi, r, tau, ap, bp, Hc)
    Ap, Bp = inner_coeffs(phi, r, tau, ap, bp, Hp)

    print()
    print("Call at boundary candidate:")
    print(
       continuation_values(
            np.array([np.log(Kc)]),
            Ac,
            Bc,
            ap,
            bp,
        )[0]
    )


    # switching boundary: Vc - Vp is itself one trig polynomial
    D = lambda x: continuation_values(x, Ac-Ap, Bc-Bp, ap, bp)[0]
    xstar = brentq(D, ap, bp, xtol=1e-12, rtol=1e-12)


    # outer coefficients: put on [a,x*], call on [x*,b]
    omega = np.arange(Nin)*np.pi/(bp-ap)
    nu    = np.arange(Nout)*np.pi/(b-a)
    H = (Ap @ build_Ic(omega, nu, a, ap, b, a, xstar)
       - Bp @ build_Is(omega, nu, a, ap, b, a, xstar)
       + Ac @ build_Ic(omega, nu, a, ap, b, xstar, b)
       - Bc @ build_Is(omega, nu, a, ap, b, xstar, b))

    w = np.ones(Nout); w[0] = 0.5
    return np.exp(-r*tc) * np.sum(w * H * np.real(phi(nu, tc)*np.exp(1j*nu*(x0-a))))


def cos_chooser_numerical(
    phi,
    cumulants,
    S0,
    Kc,
    Kp,
    r,
    tc,
    tau,
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

    tau = tau - tc
   
    # Truncation interval at choice time
  
    x0 = np.log(S0)

    c1, c2, c4 = cumulants(tc)

    a, b = truncation_interval(x0 + c1, c2, c4, L)

    # Numerical integration grid
    dx = (b - a) / M

    x = a + (np.arange(M) + 0.5) * dx

    S = np.exp(x)
    # Continuation call and put values
  
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

    # Numerical COS payoff coefficients

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

    # COS expectation

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
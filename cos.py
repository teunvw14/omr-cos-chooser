import numpy as np
import cf

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

    a, b = truncation_interval(c1, c2, c4, L)

    k = np.arange(N)

    Hk = put_coeffs(a, b, k)

    put = K * cos_value(phi=phi, x=x0, r=r, tau=tau, a=a, b=b, Hk=Hk)
    
    return put

def cos_call_price(phi, cumulants, S0, K, r, tau, N, L):
    put = cos_put_price(phi=phi, cumulants=cumulants, S0=S0, K=K, r=r, tau=tau, N=N, L=L)

    call = put + S0 - K * np.exp(-r * tau)

    return call
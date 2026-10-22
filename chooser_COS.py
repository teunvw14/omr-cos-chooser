
import numpy as np
import cf


def Chi_Psi(a, b, c, d, k):
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


def PutCoefficients(a, b, k):
    c = a
    d = 0.0

    chi, psi = Chi_Psi(a, b, c, d, k)

    H_k = 2.0 / (b - a) * (psi - chi)

    return H_k


def cos_put_price(phi, S0, K, r, tau, N, L):
    # Log-moneyness
    x0 = np.log(S0 / K)

    # Truncation domain - To be updated with use of cumulants?
    a = -L * np.sqrt(tau)
    b = L * np.sqrt(tau)

    # COS frequencies
    k = np.arange(N)
    u = k * np.pi / (b - a)

    # Put payoff coefficients
    H_k = PutCoefficients(a, b, k)

    # COS expansion terms
    temp = phi(u, tau) * H_k
    temp[0] *= 0.5

    put = (
        np.exp(-r * tau)
        * K
        * np.real(
            np.sum(
                np.exp(1j * u * (x0 - a)) * temp
            )
        )
    )

    return float(put)


def cos_call_put_price(phi, S0, K, r, tau, N, L):
    put = cos_put_price(phi, S0, K, r, tau, N, L)

    call = put + S0 - K * np.exp(-r * tau)

    return call, put

def cos_chooser_price(phi, S0, K, r, tau, tc, N, L):
    if not 0.0 <= tc <= tau:
        raise ValueError(
            "choice_time must satisfy 0 <= choice_time <= tau."
        )

    # Call with strike K and final maturity tau
    call, _ = cos_call_put_price(
        phi=phi,
        S0=S0,
        K=K,
        r=r,
        tau=tau,
        N=N,
        L=L,
    )

    # Put with adjusted strike and maturity tc
    adjusted_K = K * np.exp(-r * (tau - tc))

    _, put = cos_call_put_price(
        phi=phi,
        S0=S0,
        K=adjusted_K,
        r=r,
        tau=tc,
        N=N,
        L=L,
    )

    chooser = call + put

    return chooser


def main():
    S0 = 100
    r = 0.05
    K = 110
    tau = 10

    sigma = 0.15
    muJ = -0.05
    sigmaJ = 0.3
    xiP = 0.7

    N = 256
    L = 10

    phi_merton = cf.Merton(
        r,
        sigma,
        xiP,
        muJ,
        sigmaJ,   
    )

    call, put = cos_call_put_price(
        phi=phi_merton,
        S0=S0,
        K=K,
        r=r,
        tau=tau,
        N=N,
        L=L,
    )
    print("Merton model parameters:")
    print("r:", r)
    print("muJ:", muJ)
    print("sigmaJ:", sigmaJ)
    print("sigma:", sigma)
    print("xiP:", xiP)
    print("------------------------")
    print("COS parameters:")
    print("N:", N)
    print("L:", L)
    print("------------------------")
    print("Option parameters:")    
    print("Maturity:", tau)
    print("Spot:", S0)
    print("Strike:", K)
    print("------------------------")
    print("Call:", call)
    print("Put: ", put)
    print("------------------------")
    tc = 5.0
    chooser = cos_chooser_price(
        phi =phi_merton,
        S0=S0,
        K=K,
        r=r,
        tau=tau,
        tc=tc,
        N=N,
        L=L,
    )
    print("Choice time:", tc)
    print("Chooser:", chooser)
#####################################################

    xiP = 0.7
    p1 = 0.4
    alpha1 = 10.0
    alpha2 = 5.0

    phi_kou = cf.Kou(
        r,
        sigma,
        xiP,
        p1,
        alpha1,
        alpha2,
    )

    call, put = cos_call_put_price(
            phi=phi_kou,
            S0=S0,
            K=K,
            r=r,
            tau=tau,
            N=N,
            L=L,
    )
    print("------------------------")
    print("Kou model parameters:")
    print("r:", r)    
    print("sigma:", sigma)
    print("xiP:", xiP)
    print("p1:", p1)
    print("alpha1:", alpha1)
    print("alpha2:", alpha2)
    print("------------------------")
    print("COS parameters:")
    print("N:", N)
    print("L:", L)
    print("------------------------")
    print("Option parameters:")    
    print("Maturity:", tau)
    print("Spot:", S0)
    print("Strike:", K)
    print("------------------------")
    print("Call:", call)
    print("Put: ", put)
    print("------------------------")
    tc = 5.0
    chooser = cos_chooser_price(
        phi =phi_kou,
        S0=S0,
        K=K,
        r=r,
        tau=tau,
        tc=tc,
        N=N,
        L=L,
    )
    print("Choice time:", tc)
    print("Chooser:", chooser)
if __name__ == "__main__":
    main()

import numpy as np


def ChFForMertonModel(r, muJ, sigmaJ, sigma, xiP):

    helpExp = np.exp(muJ + 0.5 * sigmaJ**2) - 1.0

    def cf(tau):

        return lambda u: np.exp(
            1j * u * (r - xiP * helpExp - 0.5 * sigma**2) * tau
            - 0.5 * sigma**2 * u**2 * tau
            + xiP * tau
            * (np.exp(1j * u * muJ - 0.5 * sigmaJ**2 * u**2) - 1.0)
        )

    return cf



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


def cos_put_price(cf, S0, K, r, tau, N, L):
    # Log-moneyness
    x0 = np.log(S0 / K)

    # Truncation domain
    a = -L * np.sqrt(tau)
    b = L * np.sqrt(tau)

    # COS frequencies
    k = np.arange(N)
    u = k * np.pi / (b - a)

    # Put payoff coefficients
    H_k = PutCoefficients(a, b, k)

    # COS expansion terms
    temp = cf(u) * H_k
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


def cos_call_put_price(cf, S0, K, r, tau, N, L):
    put = cos_put_price(cf, S0, K, r, tau, N, L)

    call = put + S0 - K * np.exp(-r * tau)

    return call, put

def cos_chooser_price(cf, S0, K, r, tau, tc, N, L):
    if not 0.0 <= tc <= tau:
        raise ValueError(
            "choice_time must satisfy 0 <= choice_time <= tau."
        )

    # Call with strike K and final maturity tau
    call, _ = cos_call_put_price(
        cf=cf(tau),
        S0=S0,
        K=K,
        r=r,
        tau=tau,
        N=N,
        L=L,
    )

    # Put with adjusted strike and maturity choice_time
    adjusted_K = K * np.exp(-r * (tau - tc))

    _, put = cos_call_put_price(
        cf=cf(tc),
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

    N = 1000
    L = 10

    cf = ChFForMertonModel(
        r,
        muJ,
        sigmaJ,
        sigma,
        xiP,
    )

    call, put = cos_call_put_price(
        cf=cf(tau),
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
        cf =cf,
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
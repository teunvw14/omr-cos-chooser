import numpy as np

from cf import merton
from cos import cos_chooser_analytic

from mc import price_single_date
from path import merton_values


def main():

    # Contract
    S0 = 100.0
    K = 100.0

    r = 0.05
    tau = 1.0
    tc = 0.5

    # Merton model
    sigma = 0.20
    xiP = 1.0
    muJ = -0.10
    sigmaJ = 0.20

    # COS parameters
    N = 2**7
    L = 10.0

    # Characteristic function
    phi, cumulants = merton(
        r=r,
        sigma=sigma,
        xiP=xiP,
        muJ=muJ,
        sigmaJ=sigmaJ,
    )

    # COS price
    price_cos = cos_chooser_analytic(
        phi=phi,
        cumulants=cumulants,
        S0=S0,
        Kc=K,
        Kp=K,
        r=r,
        tc=tc,
        tau=tau,
        Nin=N,
        Nout=N,
        L=L,
    )

    # Monte Carlo price

    price_mc, stderr, ci95 = price_single_date(
        values_func=merton_values,
        S0=S0,
        r=r,
        tau=tau,
        tc=tc,
        Kc=K,
        Kp=K,
        n_outer=20000,
        n_inner=100,
        seed=12345,
        sigma=sigma,
        xiP=xiP,
        muJ=muJ,
        sigmaJ=sigmaJ,
    )

    # Output

    print()
    print("CHOOSER VALIDATION")
    print("==================")
    print()

    print(f"COS price : {price_cos:.6f}")
    print()

    print(f"MC price  : {price_mc:.6f}")
    print(f"MC SE     : {stderr:.6f}")
    print(
        f"MC 95% CI : "
        f"[{ci95[0]:.6f}, {ci95[1]:.6f}]"
    )

    print()

    inside = (
        ci95[0]
        <= price_cos
        <= ci95[1]
    )

    print(
        f"COS inside MC confidence interval? "
        f"{inside}"
    )


if __name__ == "__main__":
    main()

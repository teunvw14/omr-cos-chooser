
import cf
from cos import cos_chooser_price
from mc import price_single_date
from path import merton_values
import numpy as np
import matplotlib.pyplot as plt


S0 = 100.0
Kc = 100.0
Kp = 100.0
r = 0.05

tc = 0.5
T = 1.0

sigma = 0.15
xiP = 0.10
muJ = -0.10
sigmaJ = 0.20
q = 0.0

N = 256
L = 10
M = 2000

n_outer = 200_000
n_inner = 500


phi, cumulants = cf.merton(
    r=r,
    sigma=sigma,
    xiP=xiP,
    muJ=muJ,
    sigmaJ=sigmaJ,
    q=q,
)


# COS
cos_price = cos_chooser_price(
    phi=phi,
    cumulants=cumulants,
    S0=S0,
    Kc=Kc,
    Kp=Kp,
    r=r,
    tc=tc,
    T=T,
    N=N,
    L=L,
    M=M,
)


# Monte Carlo
mc_price, se, ci95 = price_single_date(
    values_func=merton_values,
    S0=S0,
    r=r,
    T=T,
    tc=tc,
    Kc=Kc,
    Kp=Kp,
    n_outer=n_outer,
    n_inner=n_inner,
    seed=1234,
    sigma=sigma,
    xiP=xiP,
    muJ=muJ,
    sigmaJ=sigmaJ,
    q=q,
)


print(f"COS      = {cos_price:.8f}")
print(f"MC       = {mc_price:.8f}")
print(f"SE       = {se:.8f}")
print(f"95% CI   = ({ci95[0]:.8f}, {ci95[1]:.8f})")
print(f"ABS ERR  = {abs(cos_price - mc_price):.8f}")
print("\n")


# COS convergence
N_values = np.arange(16, 128+1, 4)
prices = []

for N in N_values:

    price = cos_chooser_price(
        phi=phi,
        cumulants=cumulants,
        S0=S0,
        Kc=Kc,
        Kp=Kp,
        r=r,
        tc=tc,
        T=T,
        N=N,
        L=L,
        M=M,
    )

    prices.append(price)


plt.figure(figsize=(8, 5))

plt.plot(
    N_values,
    prices,
    "o-r",
    linewidth=2,
    label="COS",
)

plt.axhline(
    mc_price,
    color="blue",
    linewidth=2,
    label="MC",
)

plt.axhline(
    ci95[0],
    color="blue",
    linestyle="--",
    linewidth=1,
)

plt.axhline(
    ci95[1],
    color="blue",
    linestyle="--",
    linewidth=1,
    label="MC 95% CI",
)

plt.xlabel("Number of COS Terms (N)")
plt.ylabel("Chooser Price")
plt.title(
    f"COS chooser under Merton vs Monte Carlo "
    f"(n_outer={n_outer:,}, n_inner={n_inner:,})"
)
plt.grid(True)
plt.legend()

plt.show()

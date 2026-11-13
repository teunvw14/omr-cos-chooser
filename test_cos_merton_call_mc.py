import cf
from cos import cos_call_price
from mc import price_call
from path import merton_values
import numpy as np
import matplotlib.pyplot as plt


S0 = 100.0
K = 100.0
r = 0.05
tau = 1.0

sigma = 0.20
xiP = 0.50
muJ = -0.10
sigmaJ = 0.25

N = 256
L = 10

n_paths = 10_000_000


phi, cumulants = cf.merton(
    r=r,
    sigma=sigma,
    xiP=xiP,
    muJ=muJ,
    sigmaJ=sigmaJ,
)

cos_price = cos_call_price(
    phi=phi,
    cumulants=cumulants,
    S0=S0,
    K=K,
    r=r,
    tau=tau,
    N=N,
    L=L,
)

mc_price, se, ci95 = price_call(
    values_func=merton_values,
    S0=S0,
    K=K,
    r=r,
    tau=tau,
    n_paths=n_paths,
    seed=1234,
    sigma=sigma,
    xiP=xiP,
    muJ=muJ,
    sigmaJ=sigmaJ,
)

print(f"COS      = {cos_price:.8f}")
print(f"MC       = {mc_price:.8f}")
print(f"SE       = {se:.8f}")
print(f"95% CI   = ({ci95[0]:.8f}, {ci95[1]:.8f})")
print(f"ABS ERR  = {abs(cos_price - mc_price):.8f}")
print("\n")



N_values = np.arange(16, 128+1, 4)
prices = []

for N in N_values:

    price = cos_call_price(
        phi=phi,
        cumulants=cumulants,
        S0=S0,
        K=K,
        r=r,
        tau=tau,
        N=N,
        L=L,
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
plt.ylabel("Call Price")
plt.title(f"COS call under Merton vs Monte Carlo (n_paths={n_paths:,})")
plt.grid(True)
plt.legend()

plt.show()
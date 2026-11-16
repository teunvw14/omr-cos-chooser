import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

import cf

from cos import (
    cos_chooser_analytic_two_dates,
)

from mc import price_two_dates


# ==================================================
# Contract parameters
# ==================================================

S0 = 100.0

# Alternatives available at tc1
Kc1 = 95.0
Kp1 = 105.0

# Alternatives available after waiting until tc2
Kc2 = 105.0
Kp2 = 95.0

r = 0.05
sigma = 0.20
q = 0.0

tc1 = 0.25
tc2 = 0.50
T = 1.00

L = 10


# ==================================================
# Monte Carlo parameters
# ==================================================

n_outer = 50_000
n_inner = 500
seed = 1234


# ==================================================
# Model
# ==================================================

phi, cumulants = cf.gbm(
    r=r,
    sigma=sigma,
    q=q,
)


# ==================================================
# High-resolution COS price
# ==================================================

N_reference = 256

cos_price = cos_chooser_analytic_two_dates(
    phi=phi,
    cumulants=cumulants,
    S0=S0,
    Kc1=Kc1,
    Kp1=Kp1,
    Kc2=Kc2,
    Kp2=Kp2,
    r=r,
    tc1=tc1,
    tc2=tc2,
    T=T,
    Nin=N_reference,
    Nout=N_reference,
    L=L,
)


# ==================================================
# Nested Monte Carlo price
# ==================================================

mc_price, se, ci95 = price_two_dates(
    S0=S0,
    Kc1=Kc1,
    Kp1=Kp1,
    Kc2=Kc2,
    Kp2=Kp2,
    r=r,
    sigma=sigma,
    q=q,
    tc1=tc1,
    tc2=tc2,
    T=T,
    n_outer=n_outer,
    n_inner=n_inner,
    seed=seed,
)


# ==================================================
# Print reference comparison
# ==================================================

print()
print(f"COS      = {cos_price:.8f}")
print(f"MC       = {mc_price:.8f}")
print(f"SE       = {se:.8f}")

print(
    f"95% CI   = "
    f"({ci95[0]:.8f}, {ci95[1]:.8f})"
)

print(
    f"ABS ERR  = "
    f"{abs(cos_price - mc_price):.8f}"
)

print(
    f"Z-SCORE  = "
    f"{(cos_price - mc_price) / se:.4f}"
)

print()


# ==================================================
# COS convergence
# ==================================================

N_values = np.arange(
    16,
    129,
    4,
)

cos_prices = []

for N in N_values:

    price = cos_chooser_analytic_two_dates(
        phi=phi,
        cumulants=cumulants,
        S0=S0,
        Kc1=Kc1,
        Kp1=Kp1,
        Kc2=Kc2,
        Kp2=Kp2,
        r=r,
        tc1=tc1,
        tc2=tc2,
        T=T,
        Nin=N,
        Nout=N,
        L=L,
    )

    cos_prices.append(price)

cos_prices = np.asarray(
    cos_prices
)


# ==================================================
# Plot
# ==================================================

plt.figure(
    figsize=(9, 5.5)
)

plt.plot(
    N_values,
    cos_prices,
    "o-r",
    linewidth=2,
    markersize=4,
    label="Analytic COS",
)

plt.axhline(
    mc_price,
    color="blue",
    linewidth=2,
    label="Nested Monte Carlo",
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
)

plt.fill_between(
    N_values,
    ci95[0],
    ci95[1],
    color="blue",
    alpha=0.10,
    label="confidence interval",
)

plt.xlabel(
    "Number of COS terms (N)"
)

plt.ylabel(
    "Two-date chooser price"
)

plt.title(
    "Two-date analytic COS chooser vs nested Monte Carlo\n"
    f"tc1={tc1}, tc2={tc2}, T={T}, "
    f"outer={n_outer:,}, inner={n_inner:,}"
)

plt.grid(
    True,
    alpha=0.35,
)

plt.legend()

ax = plt.gca()

# Choose a readable vertical scale automatically.
all_reference_values = np.concatenate([
    cos_prices,
    np.array([
        mc_price,
        ci95[0],
        ci95[1],
    ]),
])

y_min = np.min(
    all_reference_values
)

y_max = np.max(
    all_reference_values
)

padding = max(
    0.05,
    0.15 * (y_max - y_min),
)

ax.set_ylim(
    y_min - padding,
    y_max + padding,
)

ax.yaxis.set_major_locator(
    MultipleLocator(0.05)
)

plt.tight_layout()
plt.show()
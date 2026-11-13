import numpy as np

import cf

from cos import (
    put_coeffs,
    truncation_interval,
    cos_put_price,
)

from coefficients import (
    inner_coeffs,
    continuation_values,
)

S0 = 100.0
K = 100.0

r = 0.05
sigma = 0.20

tau = 0.50

N = 256
L = 10


phi, cumulants = cf.gbm(
    r=r,
    sigma=sigma,
)

c1, c2, c4 = cumulants(tau)

x0 = np.log(S0 / K)

a, b = truncation_interval(
    x0 + c1,
    c2,
    c4,
    L,
)


k = np.arange(N)

H_put = put_coeffs(
    a,
    b,
    k,
)


A_put, B_put = inner_coeffs(
    phi=phi,
    r=r,
    tau=tau,
    a=a,
    b=b,
    H=H_put,
)


x_grid = np.linspace(
    a,
    b,
    100,
)

cos_values = K *continuation_values(
    x_grid,
    A_put,
    B_put,
    a,
    b,
)


reference_values = np.empty_like(
    x_grid
)

for i, x in enumerate(x_grid):

    spot = K * np.exp(x)

    reference_values[i] = cos_put_price(
        phi=phi,
        cumulants=cumulants,
        S0=spot,
        K=K,
        r=r,
        tau=tau,
        N=N,
        L=L,
    )


errors = np.abs(
    cos_values
    - reference_values
)

print()

print(
    f"Maximum error : "
    f"{np.max(errors):.6e}"
)

print(
    f"Mean error    : "
    f"{np.mean(errors):.6e}"
)

print()

print(
    "Continuation value at x0"
)

center = np.argmin(
    np.abs(x_grid - x0)
)

print(
    f"Expansion : "
    f"{cos_values[center]:.12f}"
)

print(
    f"Reference : "
    f"{reference_values[center]:.12f}"
)

print(
    f"Error     : "
    f"{errors[center]:.6e}"
)

idx = np.argmax(errors)

print()
print("Worst point")
print(f"x      = {x_grid[idx]:.12f}")
print(f"COS    = {cos_values[idx]:.12f}")
print(f"REF    = {reference_values[idx]:.12f}")
print(f"ERROR  = {errors[idx]:.12f}")

import matplotlib.pyplot as plt

plt.plot(
    x_grid,
    cos_values,
    label="Expansion",
)

plt.plot(
    x_grid,
    reference_values,
    "--",
    label="Reference",
)

plt.legend()
plt.show()
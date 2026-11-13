import numpy as np

import cf

from cos import (
    call_coeffs,
    truncation_interval,
    cos_call_price,
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

H_call = call_coeffs(
    a,
    b,
    k,
)


A_call, B_call = inner_coeffs(
    phi=phi,
    r=r,
    tau=tau,
    a=a,
    b=b,
    H=H_call,
)

x_grid = np.linspace(
    a,
    b,
    100,
)

cos_values = (
    K
    * continuation_values(
        x_grid,
        A_call,
        B_call,
        a,
        b,
    )
)


reference_values = np.empty_like(
    x_grid
)

for i, x in enumerate(x_grid):

    spot = K * np.exp(x)

    reference_values[i] = cos_call_price(
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

print()

idx = np.argmax(errors)

print(
    "Worst point"
)

print(
    f"x      = "
    f"{x_grid[idx]:.12f}"
)

print(
    f"COS    = "
    f"{cos_values[idx]:.12f}"
)

print(
    f"REF    = "
    f"{reference_values[idx]:.12f}"
)

print(
    f"ERROR  = "
    f"{errors[idx]:.12f}"
)
import numpy as np

import cf

from cos import (
    put_coeffs,
    call_coeffs,
    chooser_coeffs,
    chooser_boundary,
    truncation_interval,
)

from integrals import (
    frequencies,
)

from coefficients import (
    inner_coeffs,
    continuation_values,
)


# --------------------------------------------------
# Parameters
# --------------------------------------------------

S0 = 100.0

Kc = 100.0
Kp = 100.0

r = 0.05
sigma = 0.20

tc = 0.50
T = 1.00

tau = T - tc

N = 256
L = 10


# --------------------------------------------------
# Model
# --------------------------------------------------

phi, cumulants = cf.gbm(
    r=r,
    sigma=sigma,
)

c1, c2, c4 = cumulants(tau)

x0 = np.log(S0 / Kc)

a, b = truncation_interval(
    x0 + c1,
    c2,
    c4,
    L,
)

omega = frequencies(
    N,
    a,
    b,
)

nu = frequencies(
    N,
    a,
    b,
)


# --------------------------------------------------
# Boundary
# --------------------------------------------------

x_star = chooser_boundary(
    phi=phi,
    cumulants=cumulants,
    r=r,
    tau=tau,
    N=N,
    L=L,
    Kc=Kc,
    Kp=Kp,
)

print()
print(f"x_star = {x_star:.12f}")


# --------------------------------------------------
# Payoff coefficients
# --------------------------------------------------

k = np.arange(N)

H_put = put_coeffs(
    a,
    b,
    k,
)

H_call = call_coeffs(
    a,
    b,
    k,
)


# --------------------------------------------------
# Continuation coefficients
# --------------------------------------------------

A_put, B_put = inner_coeffs(
    phi=phi,
    r=r,
    tau=tau,
    a=a,
    b=b,
    H=H_put,
)

A_call, B_call = inner_coeffs(
    phi=phi,
    r=r,
    tau=tau,
    a=a,
    b=b,
    H=H_call,
)


# --------------------------------------------------
# Chooser coefficients
# --------------------------------------------------

H_chooser = chooser_coeffs(
    A_call,
    B_call,
    A_put,
    B_put,
    omega,
    nu,
    a,
    a,
    b,
    x_star,
)


# --------------------------------------------------
# Project chooser coeffs
# from time tc back to time 0
# --------------------------------------------------

A_chooser, B_chooser = inner_coeffs(
    phi=phi,
    r=r,
    tau=tc,
    a=a,
    b=b,
    H=H_chooser,
)


# --------------------------------------------------
# Evaluate on a grid
# --------------------------------------------------

x_grid = np.linspace(
    a,
    b,
    500,
)

call_values = (
    Kc
    * continuation_values(
        x_grid,
        A_call,
        B_call,
        a,
        b,
    )
)

put_values = (
    Kp
    * continuation_values(
        x_grid,
        A_put,
        B_put,
        a,
        b,
    )
)

reference_values = np.maximum(
    call_values,
    put_values,
)

chooser_values = continuation_values(
    x_grid,
    A_chooser,
    B_chooser,
    a,
    b,
)


# --------------------------------------------------
# Errors
# --------------------------------------------------

errors = np.abs(
    chooser_values
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

idx = np.argmax(errors)

print()

print("Worst point")

print(
    f"x      = "
    f"{x_grid[idx]:.12f}"
)

print(
    f"Chooser = "
    f"{chooser_values[idx]:.12f}"
)

print(
    f"Max(C,P)= "
    f"{reference_values[idx]:.12f}"
)

print(
    f"Error   = "
    f"{errors[idx]:.12f}"
)


# --------------------------------------------------
# Boundary sanity check
# --------------------------------------------------

boundary_idx = np.argmin(
    np.abs(
        x_grid - x_star
    )
)

print()

print("Near boundary")

print(
    f"x = {x_grid[boundary_idx]:.12f}"
)

print(
    f"Call = {call_values[boundary_idx]:.12f}"
)

print(
    f"Put  = {put_values[boundary_idx]:.12f}"
)

print(
    f"Chooser = {chooser_values[boundary_idx]:.12f}"
)
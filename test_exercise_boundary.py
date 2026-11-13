import numpy as np

import cf
from cos import chooser_boundary, cos_call_price, cos_put_price


# Contract parameters
call_strike = 100.0
put_strike = 100.0

rate = 0.05
volatility = 0.20

maturity = 1.0
choice_time = 0.50

# COS parameters
number_of_terms = 256
interval_size = 10

# Time from choice date to maturity
remaining_time = maturity - choice_time


# Build the GBM characteristic function and cumulants
phi, cumulants = cf.gbm(
    r=rate,
    sigma=volatility,
)


# Numerical boundary from the COS prices
boundary_log_price = chooser_boundary(
    phi=phi,
    cumulants=cumulants,
    r=rate,
    tau=remaining_time,
    N=number_of_terms,
    L=interval_size,
    Kc=call_strike,
    Kp=put_strike,
)

boundary_spot = np.exp(boundary_log_price)


# Exact GBM boundary for equal call and put strikes
exact_boundary_spot = (
    call_strike
    * np.exp(-rate * remaining_time)
)

exact_boundary_log_price = np.log(
    exact_boundary_spot
)


# Evaluate the call and put at the numerical boundary
call_value = cos_call_price(
    phi=phi,
    cumulants=cumulants,
    S0=boundary_spot,
    K=call_strike,
    r=rate,
    tau=remaining_time,
    N=number_of_terms,
    L=interval_size,
)

put_value = cos_put_price(
    phi=phi,
    cumulants=cumulants,
    S0=boundary_spot,
    K=put_strike,
    r=rate,
    tau=remaining_time,
    N=number_of_terms,
    L=interval_size,
)


# Results
print("CHOOSER BOUNDARY UNDER GBM")
print()

print("Log-price boundary")
print(f"COS boundary   = {boundary_log_price:.12f}")
print(f"Exact boundary = {exact_boundary_log_price:.12f}")
print(
    "Absolute error = "
    f"{abs(boundary_log_price - exact_boundary_log_price):.3e}"
)
print()

print("Spot-price boundary")
print(f"COS boundary   = {boundary_spot:.12f}")
print(f"Exact boundary = {exact_boundary_spot:.12f}")
print(
    "Absolute error = "
    f"{abs(boundary_spot - exact_boundary_spot):.3e}"
)
print()

print("Values at the numerical boundary")
print(f"Call value     = {call_value:.12f}")
print(f"Put value      = {put_value:.12f}")
print(f"Call minus put = {call_value - put_value:.3e}")
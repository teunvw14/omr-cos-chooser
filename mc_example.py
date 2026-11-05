import numpy as np
from mc import price_two_dates
from path import merton_values



price, stderr, ci95 = price_two_dates(
    values_func=merton_values,
    S0=100.0,
    r=0.05,
    T=1.0,
    t1=0.3,
    t2=0.6,
    Kc=105.0,
    Kp=95.0,
    n_outer=1000,
    n_inner_t1=100,
    n_inner_t2=50,
    seed=12345,
    sigma=0.2,
    xiP=1.0,
    muJ=-0.1,
    sigmaJ=0.2,
)

print(f"Price : {price:.6f}")
print(f"SE    : {stderr:.6f}")
print(f"95% CI: [{ci95[0]:.6f}, {ci95[1]:.6f}]")
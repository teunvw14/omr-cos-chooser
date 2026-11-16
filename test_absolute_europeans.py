import numpy as np
import cf

from cos import (
    cos_call_price,
    cos_put_price,
)

S0 = 100.0
K = 100.0

r = 0.05
sigma = 0.20

T = 1.0

N = 256
L = 10

phi, cumulants = cf.gbm(
    r=r,
    sigma=sigma,
)

call = cos_call_price(
    phi=phi,
    cumulants=cumulants,
    S0=S0,
    K=K,
    r=r,
    tau=T,
    N=N,
    L=L,
)

put = cos_put_price(
    phi=phi,
    cumulants=cumulants,
    S0=S0,
    K=K,
    r=r,
    tau=T,
    N=N,
    L=L,
)

parity_error = (
    call
    - put
    - (
        S0
        - K * np.exp(-r*T)
    )
)

print()
print("Call =", call)
print("Put  =", put)
print()
print("Put-call parity error =", parity_error)
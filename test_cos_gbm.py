import numpy as np
import cf
from analytical import bs_call, bs_put
from cos import cos_call_price, cos_put_price
import matplotlib.pyplot as plt


# Parameters
S0 = 100.0
K = 100.0
r = 0.05
sigma = 0.20
tau = 1.0

N = 256
L = 10


# COS model
phi, cumulants = cf.gbm(r=r, sigma=sigma)


# COS prices
cos_call = cos_call_price(
    phi=phi,
    cumulants=cumulants,
    S0=S0,
    K=K,
    r=r,
    tau=tau,
    N=N,
    L=L,
)

cos_put = cos_put_price(
    phi=phi,
    cumulants=cumulants,
    S0=S0,
    K=K,
    r=r,
    tau=tau,
    N=N,
    L=L,
)


# Black-Scholes prices
bs_call_price = bs_call(
    S0=S0,
    K=K,
    r=r,
    tau=tau,
    sigma=sigma,
)

bs_put_price = bs_put(
    S0=S0,
    K=K,
    r=r,
    tau=tau,
    sigma=sigma,
)


print("CALL")
print(f"COS : {cos_call:.10f}")
print(f"BS  : {bs_call_price:.10f}")
print(f"ERR : {abs(cos_call - bs_call_price):.3e}")
print()

print("PUT")
print(f"COS : {cos_put:.10f}")
print(f"BS  : {bs_put_price:.10f}")
print(f"ERR : {abs(cos_put - bs_put_price):.3e}")
print()

print("PUT-CALL PARITY")
print(
    f"{cos_call - cos_put:.10f}"
)
print(
    f"{S0 - K*np.exp(-r*tau):.10f}"
)

N_values = np.arange(16, 128+1,4)
errors = []

for N in N_values:

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

    err = abs(cos_price - bs_call_price)

    # avoid log(0)
    errors.append(max(err, 1e-16))

    print(
        f"N={N:4d}  "
        f"price={cos_price:.12f}  "
        f"error={err:.3e}"
    )

plt.figure(figsize=(7, 4))

plt.semilogy(N_values, errors, marker="o")

plt.xlabel("N")
plt.ylabel("Absolute Error")
plt.title("COS Convergence under GBM against Analytical Black-Scholes")

plt.grid(True, which="both")

plt.show()
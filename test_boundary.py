import numpy as np
import cf

from cos import chooser_boundary

r = 0.05
sigma = 0.20

Kc = 100.0
Kp = 100.0

t1 = 0.5
T = 1.0

tau = T - t1

phi, cumulants = cf.gbm(
    r=r,
    sigma=sigma,
)

xstar = chooser_boundary(
    phi=phi,
    cumulants=cumulants,
    r=r,
    tau=tau,
    N=256,
    L=10,
    Kc=Kc,
    Kp=Kp,
)

print()
print("x* =", xstar)
print("S* =", np.exp(xstar))
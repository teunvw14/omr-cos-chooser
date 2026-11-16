import cf

from cos import (
    chooser_analytic,
    cos_chooser_price,
)

S0 = 100.0

Kc = 100.0
Kp = 100.0

r = 0.05
sigma = 0.20

t1 = 0.5
T = 1.0

phi, cumulants = cf.gbm(
    r=r,
    sigma=sigma,
)

analytic = chooser_analytic(
    phi=phi,
    cumulants=cumulants,
    S0=S0,
    Kc=Kc,
    Kp=Kp,
    r=r,
    t1=t1,
    T=T,
    Nin=256,
    Nout=256,
)

numerical = cos_chooser_price(
    phi=phi,
    cumulants=cumulants,
    S0=S0,
    Kc=Kc,
    Kp=Kp,
    r=r,
    tc=t1,
    T=T,
    N=256,
)

print()
print("Analytic chooser  =", analytic)
print("Numerical chooser =", numerical)
print("Difference        =", analytic - numerical)
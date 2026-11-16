import cf

from cos import cos_chooser_analytic, cos_chooser_numerical

S0 = 100.0

Kc = 100.0
Kp = 100.0

r = 0.05
sigma = 0.20

tc = 0.5
tau = 1.0

phi, cumulants = cf.gbm(
    r=r,
    sigma=sigma,
)

analytic = cos_chooser_analytic(
    phi=phi,
    cumulants=cumulants,
    S0=S0,
    Kc=Kc,
    Kp=Kp,
    r=r,
    tc=tc,
    tau=tau,
    Nin=256,
    Nout=256,
)

numerical = cos_chooser_numerical(
    phi=phi,
    cumulants=cumulants,
    S0=S0,
    Kc=Kc,
    Kp=Kp,
    r=r,
    tc=tc,
    tau=tau,
    N=256,
)

print()
print("Analytic COS chooser  =", analytic)
print("Numerical COS chooser =", numerical)
print("Difference        =", analytic - numerical)
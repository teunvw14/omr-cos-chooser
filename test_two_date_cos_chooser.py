import cf

from cos import cos_chooser_analytic_two_dates


S0 = 100.0

# More attractive options if chosen early
Kc1 = 95.0
Kp1 = 105.0

# Less attractive options if waiting
Kc2 = 105.0
Kp2 = 95.0

r = 0.05
sigma = 0.20

tc1 = 0.25
tc2 = 0.50
T = 1.00

Nin = 256
Nout = 256
L = 10


phi, cumulants = cf.gbm(
    r=r,
    sigma=sigma,
)


result = cos_chooser_analytic_two_dates(
    phi=phi,
    cumulants=cumulants,
    S0=S0,
    Kc1=Kc1,
    Kp1=Kp1,
    Kc2=Kc2,
    Kp2=Kp2,
    r=r,
    tc1=tc1,
    tc2=tc2,
    T=T,
    Nin=Nin,
    Nout=Nout,
    L=L,
    grid_size=4001,
    value_tol=1e-8,
    return_details=True,
)


print()
print(
    "Two-date chooser price =",
    result["price"],
)

print()
print("Strikes at tc1:")
print("Call strike =", Kc1)
print("Put strike  =", Kp1)

print()
print("Strikes at tc2:")
print("Call strike =", Kc2)
print("Put strike  =", Kp2)

print()
print("Boundary at tc2:")
print("x* =", result["xstar_tc2"])
print("S* =", result["Sstar_tc2"])

print()
print("Winning regions at tc1:")

for left, right, index, name in result["regions_tc1"]:
    print(
        f"[{left:.12f}, {right:.12f}] "
        f"-> {name}"
    )
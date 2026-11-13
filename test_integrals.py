import numpy as np

from scipy.integrate import quad

from integrals import (
    frequencies,
    build_Ic,
    build_Is,
    build_Io,
)

# --------------------------------------------------
# Parameters
# --------------------------------------------------

N = 16

a = -5.0
b = 5.0

aprime = -4.0

x_star = 0.25

omega = frequencies(
    N,
    aprime,
    b,
)

nu = frequencies(
    N,
    a,
    b,
)


# --------------------------------------------------
# Closed-form matrices
# --------------------------------------------------

Ic = build_Ic(
    omega,
    nu,
    a,
    aprime,
    b,
    x_star,
    b
)

Is = build_Is(
    omega,
    nu,
    a,
    aprime,
    b,
    x_star,
    b
)

Io = build_Io(
    nu,
    a,
    b,
    x_star,
    b
)


# --------------------------------------------------
# Numerical references
# --------------------------------------------------

def Ic_quad(k, n):

    integrand = lambda x: (
        2.0
        / (b - a)
        * np.cos(
            omega[k] * (x - aprime)
        )
        * np.cos(
            nu[n] * (x - a)
        )
    )

    value, _ = quad(
        integrand,
        x_star,
        b,
        epsabs=1e-12,
        epsrel=1e-12,
    )

    return value


def Is_quad(k, n):

    integrand = lambda x: (
        2.0
        / (b - a)
        * np.sin(
            omega[k] * (x - aprime)
        )
        * np.cos(
            nu[n] * (x - a)
        )
    )

    value, _ = quad(
        integrand,
        x_star,
        b,
        epsabs=1e-12,
        epsrel=1e-12,
    )

    return value


def Io_quad(n):

    integrand = lambda x: (
        2.0
        / (b - a)
        * np.cos(
            nu[n] * (x - a)
        )
    )

    value, _ = quad(
        integrand,
        x_star,
        b,
        epsabs=1e-12,
        epsrel=1e-12,
    )

    return value


# --------------------------------------------------
# Check I_o
# --------------------------------------------------

print("\nI_o")

for n in range(min(8, N)):

    exact = Io[n]
    ref = Io_quad(n)

    err = abs(exact - ref)

    print(
        f"n={n:2d}  "
        f"error={err:.3e}"
    )


# --------------------------------------------------
# Check I_c
# --------------------------------------------------

print("\nI_c")

max_err = 0.0

for k in range(min(8, N)):
    for n in range(min(8, N)):

        exact = Ic[k, n]
        ref = Ic_quad(k, n)

        err = abs(exact - ref)

        max_err = max(
            max_err,
            err,
        )

print(
    f"max error = {max_err:.3e}"
)


# --------------------------------------------------
# Check I_s
# --------------------------------------------------

print("\nI_s")

max_err = 0.0

for k in range(min(8, N)):
    for n in range(min(8, N)):

        exact = Is[k, n]
        ref = Is_quad(k, n)

        err = abs(exact - ref)

        max_err = max(
            max_err,
            err,
        )

print(
    f"max error = {max_err:.3e}"
)
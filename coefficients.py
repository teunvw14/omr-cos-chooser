import numpy as np


def inner_coeffs(
    phi,
    r,
    tau,
    a,
    b,
    H,
):

    k = np.arange(len(H))

    omega = (
        k
        * np.pi
        / (b - a)
    )

    phi_k = phi(
        omega,
        tau,
    )

    discount = np.exp(
        -r * tau
    )

    A = (
        discount
        * np.real(phi_k)
        * H
    )

    B = (
        discount
        * np.imag(phi_k)
        * H
    )

    return A, B

def continuation_values(
    x,
    A,
    B,
    a,
    b,
):

    x = np.atleast_1d(x)

    k = np.arange(len(A))

    omega = (
        k
        * np.pi
        / (b - a)
    )

    cos_part = np.cos(
        np.outer(
            x - a,
            omega,
        )
    )

    sin_part = np.sin(
        np.outer(
            x - a,
            omega,
        )
    )

    coeffs = (
        A[None, :]
        * cos_part
        -
        B[None, :]
        * sin_part
    )

    coeffs[:, 0] *= 0.5

    values = np.sum(
        coeffs,
        axis=1,
    )

    return values

def backward_step(
    A,
    B,
    K,
    Ic,
    Is,
    Io,
):

    return (
        A @ Ic
        - B @ Is
        - K * Io
    )
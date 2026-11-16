import numpy as np

def frequencies(N, a, b):
    k = np.arange(N)
    return k * np.pi / (b - a)

def build_Io(
    nu,
    a,
    b,
    lower,
    upper
):
    Io = np.empty_like(nu)

    Io[0] = (
        2.0
        * (upper - lower)
        / (b - a)
    )

    if len(nu) > 1:
        Io[1:] = (
            2.0
            / (b - a)
            * (
                np.sin(
                    nu[1:] * (upper - a)
                )
                -
                np.sin(
                    nu[1:] * (lower - a)
                )
            )
            / nu[1:]
        )

    return Io

def build_Ic(
    omega,
    nu,
    a,
    aprime,
    b,
    lower,
    upper,
    tol=1e-12,
):

    wk = omega[:, None]
    vn = nu[None, :]

    Ic = np.zeros(
        (len(omega), len(nu))
    )

    #
    # Generic case
    #
    mask_general = (
        np.abs(wk - vn) > tol
    )

    term1 = (
        np.sin(
            (wk - vn) * upper
            - (wk * aprime - vn * a)
        )
        -
        np.sin(
            (wk - vn) * lower
            - (wk * aprime - vn * a)
        )
    ) / (wk - vn)

    term2 = (
        np.sin(
            (wk + vn) * upper
            - (wk * aprime + vn * a)
        )
        -
        np.sin(
            (wk + vn) * lower
            - (wk * aprime + vn * a)
        )
    ) / (wk + vn)

    Ic[mask_general] = (
        (term1 + term2)
        / (b - a)
    )[mask_general]

    
    # Singular case 
    mask_singular = (
        np.abs(wk - vn) <= tol
    )

    rows, cols = np.where(mask_singular)

    for k, n in zip(rows, cols):

        omega_kn = omega[k]

        if abs(omega_kn) < tol:

            Ic[k, n] = (
                2.0
                * (upper - lower)
                / (b - a)
            )

        else:
            term1 = (
                0.5
                * (upper - lower)
                * np.cos(
                    omega_kn
                    * (a - aprime)
                )
            )

            term2 = (
                np.sin(
                    2.0 * omega_kn * upper
                    - omega_kn * (a + aprime)
                )
                -
                np.sin(
                    2.0 * omega_kn * lower
                    - omega_kn * (a + aprime)
                )
            ) / (
                4.0 * omega_kn
            )

            Ic[k, n] = (
                2.0
                / (b - a)
                * (term1 + term2)
            )

    return Ic

def build_Is(
    omega,
    nu,
    a,
    aprime,
    b,
    lower,
    upper,
    tol=1e-12,
):
    wk = omega[:, None]
    vn = nu[None, :]

    Is = np.zeros((len(omega), len(nu)))

    # Generic case
    mask_general = np.abs(wk - vn) > tol

    plus = (
        np.cos(
            (wk + vn) * upper
            - (wk * aprime + vn * a)
        )
        -
        np.cos(
            (wk + vn) * lower
            - (wk * aprime + vn * a)
        )
    ) / (wk + vn)

    minus = (
        np.cos(
            (wk - vn) * upper
            - (wk * aprime - vn * a)
        )
        -
        np.cos(
            (wk - vn) * lower
            - (wk * aprime - vn * a)
        )
    ) / (wk - vn)

    Is[mask_general] = (
        (-plus - minus) / (b - a)
    )[mask_general]

    # Singular case
    mask_singular = np.abs(wk - vn) <= tol

    rows, cols = np.where(mask_singular)

    for k, n in zip(rows, cols):

        w = omega[k]

        if abs(w) < tol:
            Is[k, n] = 0.0

        else:

            constant = (
                (upper - lower)
                * np.sin(w * (a - aprime))
            )

            oscillatory = (
                np.cos(
                    2.0 * w * lower
                    - w * (a + aprime)
                )
                -
                np.cos(
                    2.0 * w * upper
                    - w * (a + aprime)
                )
            ) / (2.0 * w)

            Is[k, n] = (
                constant + oscillatory
            ) / (b - a)

    return Is

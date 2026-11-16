import numpy as np
from scipy.optimize import brentq
from integrals import build_Ic, build_Is
from coefficients import inner_coeffs, continuation_values

def chi_psi(a, b, c, d, k):
    omega = k * np.pi / (b - a)

    # Psi coefficients
    psi = np.empty_like(omega)
    psi[0] = d - c
    psi[1:] = (
        np.sin(omega[1:] * (d - a))
        - np.sin(omega[1:] * (c - a))
    ) / omega[1:]

    # Chi coefficients
    chi = (
        np.cos(omega * (d - a)) * np.exp(d)
        - np.cos(omega * (c - a)) * np.exp(c)
        + omega * np.sin(omega * (d - a)) * np.exp(d)
        - omega * np.sin(omega * (c - a)) * np.exp(c)
    ) / (1.0 + omega**2)

    return chi, psi

def truncation_interval(c1, c2, c4, L):
    width = L * np.sqrt(c2 + np.sqrt(c4))
    a = c1 - width
    b = c1 + width

    return a, b

def put_coeffs(K,a, b, k): # Strike is passed to allow for Kc different from Kp 
    c = a
    d = np.clip(np.log(K), a, b)
    chi, psi = chi_psi(a, b, c, d, k)
    H_k = 2.0 / (b - a) * (K*psi - chi)

    return H_k

def call_coeffs(K,a, b, k):
    c = np.clip(np.log(K), a, b)
    d = b

    chi, psi = chi_psi(a, b, c, d, k,)
    H_k = (2.0 / (b - a) * (chi - K*psi)    )

    return H_k

def cos_value(phi, x, r, tau, a, b, Hk):
    k = np.arange(len(Hk))
    u = k * np.pi / (b - a)

    temp = phi(u, tau) * Hk
    temp[0] *= 0.5

    return (
        np.exp(-r * tau)
        * np.real(
            np.sum(
                np.exp(1j * u * (x - a)) * temp
            )
        )
    )

def cos_put_price(phi, cumulants, S0, K, r, tau, N, L):
    x0 = np.log(S0)

    c1, c2, c4 = cumulants(tau)

    a, b = truncation_interval(x0+c1, c2, c4, L)

    k = np.arange(N)

    Hk = put_coeffs(K,a, b, k)

    put = cos_value(phi=phi, x=x0, r=r, tau=tau, a=a, b=b, Hk=Hk)
    
    return put


def cos_call_price(phi, cumulants, S0, K, r, tau, N, L):
    put = cos_put_price(
        phi=phi,
        cumulants=cumulants,
        S0=S0,
        K=K, 
        r=r, 
        tau=tau, 
        N=N, 
        L=L)

    call = put + S0 - K * np.exp(-r * tau)

    return call

def chooser_boundary(
    phi,
    cumulants,
    r,
    tau,
    N,
    L,
    Kc,
    Kp,
    xmin=None,
    xmax=None,
):

    if xmin is None:
        xmin = np.log(min(Kc, Kp)) - 5.0

    if xmax is None:
        xmax = np.log(max(Kc, Kp)) + 5.0

    def f(x):
        S = np.exp(x)

        call_value = cos_call_price(
            phi=phi,
            cumulants=cumulants,
            S0=S,
            K=Kc,
            r=r,
            tau=tau,
            N=N,
            L=L,
        )

        put_value = cos_put_price(
            phi=phi,
            cumulants=cumulants,
            S0=S,
            K=Kp,
            r=r,
            tau=tau,
            N=N,
            L=L,
        )

        return call_value - put_value

    f_left = f(xmin)
    f_right = f(xmax)

    if f_left * f_right > 0.0:
        raise ValueError(
            "Could not bracket the chooser boundary: "
            f"f(xmin)={f_left:.6e}, "
            f"f(xmax)={f_right:.6e}."
        )

    return brentq(f, xmin, xmax, xtol=1e-12, rtol=1e-12)

def project_branch(
    A,
    B,
    omega,
    nu,
    a,
    aprime,
    b,
    lower,
    upper,
):
    """
    Project one trigonometric continuation-value branch onto
    the cosine basis of the target interval [a, b].

    The source branch is

        V(x) = sum_k [
            A[k] cos(omega[k] * (x - aprime))
            - B[k] sin(omega[k] * (x - aprime))
        ]

    and this function returns the cosine coefficients

        H[n] = 2 / (b - a)
               * integral_lower^upper
                 V(x) cos(nu[n] * (x - a)) dx.
    """

    if not a <= lower <= upper <= b:
        raise ValueError(
            "Require a <= lower <= upper <= b."
        )

    if len(A) != len(B):
        raise ValueError(
            "A and B must have the same length."
        )

    if len(A) != len(omega):
        raise ValueError(
            "A, B, and omega must have the same length."
        )

    Ic = build_Ic(
        omega,
        nu,
        a,
        aprime,
        b,
        lower,
        upper,
    )

    Is = build_Is(
        omega,
        nu,
        a,
        aprime,
        b,
        lower,
        upper,
    )

    return (
        A @ Ic
        - B @ Is
    )

def chooser_coeffs(
    A_call,
    B_call,
    A_put,
    B_put,
    omega,
    nu,
    a,
    aprime,
    b,
    x_star,
):
    H_put = project_branch(
        A=A_put,
        B=B_put,
        omega=omega,
        nu=nu,
        a=a,
        aprime=aprime,
        b=b,
        lower=a,
        upper=x_star,
    )

    H_call = project_branch(
        A=A_call,
        B=B_call,
        omega=omega,
        nu=nu,
        a=a,
        aprime=aprime,
        b=b,
        lower=x_star,
        upper=b,
    )

    return H_put + H_call

def branch_values(
    x,
    A,
    B,
    a_source,
    b_source,
):
    return continuation_values(
        x,
        A,
        B,
        a_source,
        b_source,
    )

def find_winning_regions(
    value_functions,
    branch_names,
    a,
    b,
    grid_size=2001,
    root_tol=1e-12,
    value_tol=1e-8,
    preferred_branch=None,
):
    """
    Find the upper-envelope regions of several value functions.

    Values within value_tol of the maximum are treated as tied.
    If preferred_branch is supplied, that branch wins such ties.
    """

    if len(value_functions) != len(branch_names):
        raise ValueError(
            "value_functions and branch_names must have "
            "the same length."
        )

    if preferred_branch is not None:
        if preferred_branch not in branch_names:
            raise ValueError(
                f"Unknown preferred branch: {preferred_branch}"
            )

        preferred_index = branch_names.index(
            preferred_branch
        )
    else:
        preferred_index = None

    x_grid = np.linspace(
        a,
        b,
        grid_size,
    )

    values = np.vstack([
        function(x_grid)
        for function in value_functions
    ])

    maximum_values = np.max(
        values,
        axis=0,
    )

    winners = np.argmax(
        values,
        axis=0,
    )

    # Resolve numerical ties consistently.
    if preferred_index is not None:
        preferred_values = values[
            preferred_index
        ]

        tied_with_preferred = (
            maximum_values
            - preferred_values
            <= value_tol
        )

        winners[
            tied_with_preferred
        ] = preferred_index

    change_indices = np.where(
        winners[1:] != winners[:-1]
    )[0]

    # One branch wins over the complete interval.
    if len(change_indices) == 0:
        winner = int(winners[0])

        return [
            (
                a,
                b,
                winner,
                branch_names[winner],
            )
        ]

    boundaries = [a]
    region_winners = [int(winners[0])]

    for index in change_indices:
        grid_left = x_grid[index]
        grid_right = x_grid[index + 1]

        old_winner = int(winners[index])
        new_winner = int(winners[index + 1])

        old_function = value_functions[
            old_winner
        ]

        new_function = value_functions[
            new_winner
        ]

        def difference(x):
            x_array = np.array([x])

            return float(
                old_function(x_array)[0]
                - new_function(x_array)[0]
            )

        f_left = difference(grid_left)
        f_right = difference(grid_right)

        if f_left == 0.0:
            boundary = grid_left

        elif f_right == 0.0:
            boundary = grid_right

        elif f_left * f_right < 0.0:
            boundary = brentq(
                difference,
                grid_left,
                grid_right,
                xtol=root_tol,
                rtol=root_tol,
            )

        else:
            # The grid changed winner because of value_tol,
            # but there is no actual sign-changing root.
            boundary = 0.5 * (
                grid_left + grid_right
            )

        # Do not retain zero-width regions.
        if boundary > boundaries[-1] + root_tol:
            boundaries.append(boundary)
            region_winners.append(new_winner)
        else:
            region_winners[-1] = new_winner

    boundaries.append(b)

    regions = []

    for i, winner in enumerate(
        region_winners
    ):
        left = boundaries[i]
        right = boundaries[i + 1]

        if right - left <= root_tol:
            continue

        regions.append(
            (
                left,
                right,
                winner,
                branch_names[winner],
            )
        )

    return regions

def cos_chooser_analytic(phi, cumulants, S0, Kc, Kp, r, tc, tau, Nin, Nout, L=10):
    x0, tau = np.log(S0), tau - tc

    # intervals
    c1, c2, c4 = cumulants(tc)
    a, b = truncation_interval(x0 + c1, c2, c4, L)
    e1, e2, e4 = cumulants(tau)
    d = L*np.sqrt(e2 + np.sqrt(e4))
    ap, bp = a + e1 - d, b + e1 + d

    # inner expansions: same interval, so same omega for call and put
    k  = np.arange(Nin)
    Hc = call_coeffs(Kc, ap, bp, k); Hc[0] *= 0.5
    Hp = put_coeffs(Kp, ap, bp, k); Hp[0] *= 0.5
    Ac, Bc = inner_coeffs(phi, r, tau, ap, bp, Hc)
    Ap, Bp = inner_coeffs(phi, r, tau, ap, bp, Hp)

    # switching boundary: Vc - Vp is itself one trig polynomial
    D = lambda x: continuation_values(x, Ac-Ap, Bc-Bp, ap, bp)[0]
    xstar = brentq(D, ap, bp, xtol=1e-12, rtol=1e-12)

    # outer coefficients: put on [a,x*], call on [x*,b]
    omega = np.arange(Nin)*np.pi/(bp-ap)
    nu    = np.arange(Nout)*np.pi/(b-a)
    H = (Ap @ build_Ic(omega, nu, a, ap, b, a, xstar)
       - Bp @ build_Is(omega, nu, a, ap, b, a, xstar)
       + Ac @ build_Ic(omega, nu, a, ap, b, xstar, b)
       - Bc @ build_Is(omega, nu, a, ap, b, xstar, b))

    w = np.ones(Nout); w[0] = 0.5
    return np.exp(-r*tc) * np.sum(w * H * np.real(phi(nu, tc)*np.exp(1j*nu*(x0-a))))



def cos_chooser_analytic_two_dates(
    phi,
    cumulants,
    S0,
    Kc1,
    Kp1,
    Kc2,
    Kp2,
    r,
    tc1,
    tc2,
    T,
    Nin,
    Nout,
    L=10,
    grid_size=4001,
    value_tol=1e-8,
    return_details=False,
):
    """
    Analytic COS pricer for a two-choice-date chooser.

    Contract interpretation
    -----------------------
    At tc2:
        choose between a European call with strike Kc
        and a European put with strike Kp.

    At tc1:
        choose the call immediately,
        choose the put immediately,
        or wait until tc2.

    Both European options mature at T.

    Absolute log-price coordinates are used:

        x = log(S)

    Notes
    -----
    This function uses the coefficient convention already
    validated in the one-date chooser:

        1. Payoff coefficients H are normally full coefficients.
        2. Before passing H to inner_coeffs, H[0] is halved.
        3. continuation_values does not halve its zeroth term.
        4. The final time-zero COS sum halves H1[0] through weights.
    """

    if not (0.0 < tc1 < tc2 < T):
        raise ValueError(
            "Require 0 < tc1 < tc2 < T."
        )

    if S0 <= 0.0:
        raise ValueError(
            "S0 must be positive."
        )

    if min(Kc1, Kp1, Kc2, Kp2) <= 0.0:
        raise ValueError(
            "All strikes must be positive."
    )

    if Nin < 2 or Nout < 2:
        raise ValueError(
            "Nin and Nout must be at least 2."
        )

    x0 = np.log(S0)

    dt01 = tc1
    dt12 = tc2 - tc1
    dt2T = T - tc2
    dt1T = T - tc1

    # ==================================================
    # 1. State interval at tc1
    # ==================================================

    c1, c2, c4 = cumulants(dt01)

    a1, b1 = truncation_interval(
        x0 + c1,
        c2,
        c4,
        L,
    )

    # ==================================================
    # 2. State interval at tc2
    # ==================================================

    c1, c2, c4 = cumulants(dt12)

    width12 = L * np.sqrt(
        c2 + np.sqrt(c4)
    )

    a2 = a1 + c1 - width12
    b2 = b1 + c1 + width12

    # ==================================================
    # 3. Maturity interval reached from tc2
    # ==================================================

    c1, c2, c4 = cumulants(dt2T)

    width2T = L * np.sqrt(
        c2 + np.sqrt(c4)
    )

    aT2 = a2 + c1 - width2T
    bT2 = b2 + c1 + width2T

    # ==================================================
    # 4. Terminal call and put coefficients
    # ==================================================

    kT2 = np.arange(Nin)

    HcT2 = call_coeffs(
        Kc2,
        aT2,
        bT2,
        kT2,
    )

    HpT2 = put_coeffs(
        Kp2,
        aT2,
        bT2,
        kT2,
    )

    # Apply the COS prime convention exactly once
    # before constructing the continuation expansions.

    HcT2_prime = HcT2.copy()
    HpT2_prime = HpT2.copy()

    HcT2_prime[0] *= 0.5
    HpT2_prime[0] *= 0.5

    # ==================================================
    # 5. Call and put continuation values at tc2
    # ==================================================

    Ac2, Bc2 = inner_coeffs(
        phi=phi,
        r=r,
        tau=dt2T,
        a=aT2,
        b=bT2,
        H=HcT2_prime,
    )

    Ap2, Bp2 = inner_coeffs(
        phi=phi,
        r=r,
        tau=dt2T,
        a=aT2,
        b=bT2,
        H=HpT2_prime,
    )

    def call_tc2(x):
        return continuation_values(
            x,
            Ac2,
            Bc2,
            aT2,
            bT2,
        )

    def put_tc2(x):
        return continuation_values(
            x,
            Ap2,
            Bp2,
            aT2,
            bT2,
        )

    def difference_tc2(x):
        x_array = np.array([x])

        return float(
            call_tc2(x_array)[0]
            - put_tc2(x_array)[0]
        )

    left_value = difference_tc2(a2)
    right_value = difference_tc2(b2)

    if left_value * right_value > 0.0:
        raise ValueError(
            "Could not bracket the call-put boundary "
            "at tc2. "
            f"D(a2)={left_value:.6e}, "
            f"D(b2)={right_value:.6e}."
        )

    xstar2 = brentq(
        difference_tc2,
        a2,
        b2,
        xtol=1e-12,
        rtol=1e-12,
    )

    # ==================================================
    # 6. Chooser payoff coefficients at tc2
    # ==================================================

    omegaT2 = (
        np.arange(Nin)
        * np.pi
        / (bT2 - aT2)
    )

    nu2 = (
        np.arange(Nout)
        * np.pi
        / (b2 - a2)
    )

    H2 = chooser_coeffs(
        A_call=Ac2,
        B_call=Bc2,
        A_put=Ap2,
        B_put=Bp2,
        omega=omegaT2,
        nu=nu2,
        a=a2,
        aprime=aT2,
        b=b2,
        x_star=xstar2,
    )

    # ==================================================
    # 7. Propagate the tc2 chooser back to tc1
    # ==================================================

    H2_prime = H2.copy()
    H2_prime[0] *= 0.5

    Await1, Bwait1 = inner_coeffs(
        phi=phi,
        r=r,
        tau=dt12,
        a=a2,
        b=b2,
        H=H2_prime,
    )

    def wait_tc1(x):
        return continuation_values(
            x,
            Await1,
            Bwait1,
            a2,
            b2,
        )

    # ==================================================
    # 8. Direct-to-maturity interval from tc1
    # ==================================================

    c1, c2, c4 = cumulants(dt1T)

    width1T = L * np.sqrt(
        c2 + np.sqrt(c4)
    )

    aT1 = a1 + c1 - width1T
    bT1 = b1 + c1 + width1T

    # ==================================================
    # 9. Immediate call and put values at tc1
    # ==================================================

    kT1 = np.arange(Nin)

    HcT1 = call_coeffs(
        Kc1,
        aT1,
        bT1,
        kT1,
    )

    HpT1 = put_coeffs(
        Kp1,
        aT1,
        bT1,
        kT1,
    )

    HcT1[0] *= 0.5
    HpT1[0] *= 0.5

    Ac1, Bc1 = inner_coeffs(
        phi=phi,
        r=r,
        tau=dt1T,
        a=aT1,
        b=bT1,
        H=HcT1,
    )

    Ap1, Bp1 = inner_coeffs(
        phi=phi,
        r=r,
        tau=dt1T,
        a=aT1,
        b=bT1,
        H=HpT1,
    )

    def call_tc1(x):
        return continuation_values(
            x,
            Ac1,
            Bc1,
            aT1,
            bT1,
        )

    def put_tc1(x):
        return continuation_values(
            x,
            Ap1,
            Bp1,
            aT1,
            bT1,
        )

    # ==================================================
    # 10. Find the upper envelope at tc1
    # ==================================================

    branch_names = [
        "put",
        "wait",
        "call",
    ]

    value_functions = [
        put_tc1,
        wait_tc1,
        call_tc1,
    ]

    regions = find_winning_regions(
        value_functions=[
            put_tc1,
            wait_tc1,
            call_tc1,
        ],
        branch_names=[
            "put",
            "wait",
            "call",
        ],
        a=a1,
        b=b1,
        grid_size=grid_size,
        root_tol=1e-12,
        value_tol=value_tol,
        preferred_branch="wait",
    )   

    # ==================================================
    # 11. Project every winning region
    # ==================================================

    nu1 = (
        np.arange(Nout)
        * np.pi
        / (b1 - a1)
    )

    omegaT1 = (
        np.arange(Nin)
        * np.pi
        / (bT1 - aT1)
    )

    omega2 = (
        np.arange(Nout)
        * np.pi
        / (b2 - a2)
    )

    H1 = np.zeros(
        Nout,
        dtype=float,
    )

    for lower, upper, branch_index, branch_name in regions:

        if branch_name == "put":
            contribution = project_branch(
                A=Ap1,
                B=Bp1,
                omega=omegaT1,
                nu=nu1,
                a=a1,
                aprime=aT1,
                b=b1,
                lower=lower,
                upper=upper,
            )

        elif branch_name == "wait":
            contribution = project_branch(
                A=Await1,
                B=Bwait1,
                omega=omega2,
                nu=nu1,
                a=a1,
                aprime=a2,
                b=b1,
                lower=lower,
                upper=upper,
            )

        elif branch_name == "call":
            contribution = project_branch(
                A=Ac1,
                B=Bc1,
                omega=omegaT1,
                nu=nu1,
                a=a1,
                aprime=aT1,
                b=b1,
                lower=lower,
                upper=upper,
            )

        else:
            raise RuntimeError(
                f"Unknown branch: {branch_name}"
            )

        H1 += contribution

    # ==================================================
    # 12. Propagate from tc1 to time zero
    # ==================================================

    weights = np.ones(Nout)
    weights[0] = 0.5

    characteristic_part = np.real(
        phi(nu1, tc1)
        * np.exp(
            1j
            * nu1
            * (x0 - a1)
        )
    )

    price = (
        np.exp(-r * tc1)
        * np.sum(
            weights
            * H1
            * characteristic_part
        )
    )

    if return_details:
        return {
            "price": float(price),
            "xstar_tc2": float(xstar2),
            "Sstar_tc2": float(np.exp(xstar2)),
            "regions_tc1": regions,
            "interval_tc1": (a1, b1),
            "interval_tc2": (a2, b2),
            "interval_T_from_tc2": (aT2, bT2),
            "interval_T_from_tc1": (aT1, bT1),
            "H_tc2": H2,
            "H_tc1": H1,
            "strikes_tc1": {
                "call": Kc1,
                "put": Kp1,
            },
            "strikes_tc2": {
                "call": Kc2,
                "put": Kp2,
            },
        }


    return float(price)



def cos_chooser_numerical(
    phi,
    cumulants,
    S0,
    Kc,
    Kp,
    r,
    tc,
    tau,
    N=256,
    L=10,
    M=5000,
):
    """
    Numerical COS chooser pricer.

    Choice at tc between:
        Call(Kc, T)
        Put(Kp, T)

    Uses numerical midpoint integration for the
    chooser payoff coefficients.
    """

    tau = tau - tc
   
    # Truncation interval at choice time
  
    x0 = np.log(S0)

    c1, c2, c4 = cumulants(tc)

    a, b = truncation_interval(x0 + c1, c2, c4, L)

    # Numerical integration grid
    dx = (b - a) / M

    x = a + (np.arange(M) + 0.5) * dx

    S = np.exp(x)
    # Continuation call and put values
  
    payoff = np.empty(M)

    for j in range(M):

        call = cos_call_price(
            phi=phi,
            cumulants=cumulants,
            S0=S[j],
            K=Kc,
            r=r,
            tau=tau,
            N=N,
            L=L,
        )

        put = cos_put_price(
            phi=phi,
            cumulants=cumulants,
            S0=S[j],
            K=Kp,
            r=r,
            tau=tau,
            N=N,
            L=L,
        )

        payoff[j] = max(call, put)

    # Numerical COS payoff coefficients

    k = np.arange(N)

    u = k * np.pi / (b - a)

    cos_matrix = np.cos(
        np.outer(u, x - a)
    )

    Hk = (
        2.0 / (b - a)
        * dx
        * (cos_matrix @ payoff)
    )

    # COS expectation

    phi_values = phi(u, tc)

    weights = np.ones(N)
    weights[0] = 0.5

    price = np.exp(-r * tc) * np.sum(
        weights
        * Hk
        * np.real(
            phi_values * np.exp(1j * u * (x0 - a))
        )
    )

    return price
import numpy as np
from payoff import call, put
from scipy.stats import norm

import numpy as np

from payoff import call, put

def price_call(
    values_func,
    S0,
    K,
    r,
    tau,
    n_paths,
    seed=None,
    **params,
):
    """
    Monte Carlo price of a European call.

    Returns
    -------
    price
    stderr
    ci95
    """

    rng = np.random.default_rng(seed)

    ST = values_func(
        S0=S0,
        r=r,
        times=[tau],
        n_paths=n_paths,
        rng=rng,
        **params,
    )[:, -1]

    payoffs = np.exp(-r * tau) * call(ST, K)

    price = np.mean(payoffs)

    stderr = (
        np.std(payoffs, ddof=1)
        / np.sqrt(n_paths)
    )

    ci95 = (
        price - 1.96 * stderr,
        price + 1.96 * stderr,
    )

    return price, stderr, ci95



def price_single_date(
    values_func,
    S0,
    r,
    tau,
    tc,
    Kc,
    Kp,
    n_outer,
    n_inner,
    seed=None,
    **params,
):
    """
    Nested Monte Carlo for a chooser option
    with one choice date tc.

    At tc the holder chooses between:

        Call(Kc, tau)
        Put(Kp, tau)

    Returns
    -------
    price
    stderr
    ci95
    """

    if not (0.0 <= tc <= tau):
        raise ValueError(
            "Require 0 <= tc <= tau."
        )

    if seed is not None:
        np.random.seed(seed)

    #
    # Simulate S(tc)
    #

    S_tc = values_func(
        S0=S0,
        r=r,
        times=[tc],
        n_paths=n_outer,
        **params,
    )[:, 0]

    chooser_values = np.empty(n_outer)

    disc_tc_tau = np.exp(-r * (tau - tc))
    disc_0_tc = np.exp(-r * tc)

    for i, s_tc in enumerate(S_tc):

        # Inner simulation tc -> tau
        ST = values_func(
            S0=s_tc,
            r=r,
            times=[tau - tc],
            n_paths=n_inner,
            **params,
        )[:, -1]

        call_value = (
            disc_tc_tau
            * np.mean(call(ST, Kc))
        )

        put_value = (
            disc_tc_tau
            * np.mean(put(ST, Kp))
        )

        chooser_values[i] = max(
            call_value,
            put_value,
        )

    price = disc_0_tc * np.mean(
        chooser_values
    )

    stderr = (
        disc_0_tc
        * np.std(chooser_values, ddof=1)
        / np.sqrt(n_outer)
    )

    ci95 = (
        price - 1.96 * stderr,
        price + 1.96 * stderr,
    )

    return price, stderr, ci95

def price_two_dates(
    S0,
    Kc1,
    Kp1,
    Kc2,
    Kp2,
    r,
    sigma,
    q,
    tc1,
    tc2,
    T,
    n_outer=50_000,
    n_inner=500,
    batch_size=500,
    seed=1234,
):
    """
    Nested Monte Carlo pricer for the two-date chooser.

    At tc1:
        choose call with Kc1,
        choose put with Kp1,
        or wait until tc2.

    At tc2 after waiting:
        choose call with Kc2,
        or put with Kp2.

    All options mature at T.

    Returns:
        price, standard_error, confidence_interval_95
    """

    if not (0.0 < tc1 < tc2 < T):
        raise ValueError(
            "Require 0 < tc1 < tc2 < T."
        )

    rng = np.random.default_rng(seed)

    dt01 = tc1
    dt12 = tc2 - tc1
    dt2T = T - tc2
    dt1T = T - tc1

    drift = r - q

    # --------------------------------------------------
    # Black-Scholes values
    # --------------------------------------------------

    def call_value(S, K, tau):
        S = np.asarray(S, dtype=float)

        if tau <= 0.0:
            return np.maximum(S - K, 0.0)

        sqrt_tau = np.sqrt(tau)

        d1 = (
            np.log(S / K)
            + (
                drift
                + 0.5 * sigma**2
            ) * tau
        ) / (
            sigma * sqrt_tau
        )

        d2 = (
            d1
            - sigma * sqrt_tau
        )

        return (
            S
            * np.exp(-q * tau)
            * norm.cdf(d1)
            - K
            * np.exp(-r * tau)
            * norm.cdf(d2)
        )

    def put_value(S, K, tau):
        call = call_value(S, K, tau)

        return (
            call
            - S * np.exp(-q * tau)
            + K * np.exp(-r * tau)
        )

    # --------------------------------------------------
    # Outer paths to tc1 and then maturity
    # --------------------------------------------------

    Z01 = rng.standard_normal(n_outer)
    Z12 = rng.standard_normal(n_outer)
    Z2T = rng.standard_normal(n_outer)

    S1 = (
        S0
        * np.exp(
            (
                drift
                - 0.5 * sigma**2
            ) * dt01
            + sigma
            * np.sqrt(dt01)
            * Z01
        )
    )

    S2 = (
        S1
        * np.exp(
            (
                drift
                - 0.5 * sigma**2
            ) * dt12
            + sigma
            * np.sqrt(dt12)
            * Z12
        )
    )

    ST = (
        S2
        * np.exp(
            (
                drift
                - 0.5 * sigma**2
            ) * dt2T
            + sigma
            * np.sqrt(dt2T)
            * Z2T
        )
    )

    # --------------------------------------------------
    # Immediate values at tc1
    # --------------------------------------------------

    call1 = call_value(
        S1,
        Kc1,
        dt1T,
    )

    put1 = put_value(
        S1,
        Kp1,
        dt1T,
    )

    # --------------------------------------------------
    # Nested estimate of waiting value at tc1
    # --------------------------------------------------

    wait1 = np.empty(n_outer)

    for start in range(
        0,
        n_outer,
        batch_size,
    ):
        stop = min(
            start + batch_size,
            n_outer,
        )

        batch_count = stop - start

        Z_inner = rng.standard_normal(
            (
                batch_count,
                n_inner,
            )
        )

        S2_inner = (
            S1[start:stop, None]
            * np.exp(
                (
                    drift
                    - 0.5 * sigma**2
                ) * dt12
                + sigma
                * np.sqrt(dt12)
                * Z_inner
            )
        )

        call2_inner = call_value(
            S2_inner,
            Kc2,
            dt2T,
        )

        put2_inner = put_value(
            S2_inner,
            Kp2,
            dt2T,
        )

        chooser2_inner = np.maximum(
            call2_inner,
            put2_inner,
        )

        wait1[start:stop] = (
            np.exp(-r * dt12)
            * np.mean(
                chooser2_inner,
                axis=1,
            )
        )

    # --------------------------------------------------
    # Optimal choice at tc1
    #
    # 0 = put now
    # 1 = wait
    # 2 = call now
    # --------------------------------------------------

    values1 = np.column_stack([
        put1,
        wait1,
        call1,
    ])

    choice1 = np.argmax(
        values1,
        axis=1,
    )

    choose_put1 = choice1 == 0
    choose_wait1 = choice1 == 1
    choose_call1 = choice1 == 2

    # --------------------------------------------------
    # Choice at tc2 for paths that waited
    # --------------------------------------------------

    call2 = call_value(
        S2,
        Kc2,
        dt2T,
    )

    put2 = put_value(
        S2,
        Kp2,
        dt2T,
    )

    choose_call2 = call2 >= put2

    # --------------------------------------------------
    # Realized terminal payoff
    # --------------------------------------------------

    payoff = np.empty(n_outer)

    payoff[choose_put1] = np.maximum(
        Kp1 - ST[choose_put1],
        0.0,
    )

    payoff[choose_call1] = np.maximum(
        ST[choose_call1] - Kc1,
        0.0,
    )

    wait_and_call = (
        choose_wait1
        & choose_call2
    )

    wait_and_put = (
        choose_wait1
        & ~choose_call2
    )

    payoff[wait_and_call] = np.maximum(
        ST[wait_and_call] - Kc2,
        0.0,
    )

    payoff[wait_and_put] = np.maximum(
        Kp2 - ST[wait_and_put],
        0.0,
    )

    discounted_payoff = (
        np.exp(-r * T)
        * payoff
    )

    price = np.mean(discounted_payoff)

    standard_error = (
        np.std(
            discounted_payoff,
            ddof=1,
        )
        / np.sqrt(n_outer)
    )

    ci95 = (
        price - 1.96 * standard_error,
        price + 1.96 * standard_error,
    )

    return (
        float(price),
        float(standard_error),
        (
            float(ci95[0]),
            float(ci95[1]),
        ),
    )
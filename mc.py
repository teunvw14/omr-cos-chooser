import numpy as np
from payoff import call, put

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
    T,
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

        Call(Kc, T)
        Put(Kp, T)

    Returns
    -------
    price
    stderr
    ci95
    """

    if not (0.0 <= tc <= T):
        raise ValueError(
            "Require 0 <= tc <= T."
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

    disc_tc_T = np.exp(-r * (T - tc))
    disc_0_tc = np.exp(-r * tc)

    for i, s_tc in enumerate(S_tc):

        #
        # Inner simulation tc -> T
        #

        ST = values_func(
            S0=s_tc,
            r=r,
            times=[T - tc],
            n_paths=n_inner,
            **params,
        )[:, -1]

        call_value = (
            disc_tc_T
            * np.mean(call(ST, Kc))
        )

        put_value = (
            disc_tc_T
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

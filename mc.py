import numpy as np
from payoff import call, put


def price_two_dates(
    values_func,
    S0,
    r,
    T,
    t1,
    t2,
    Kc,
    Kp,
    n_outer,
    n_inner_t1,
    n_inner_t2,
    seed=None,
    **params,
):
    """
    Nested Monte Carlo for a chooser option
    with two choice dates t1 < t2 < T.

    At t1 the holder can:
        - choose call
        - choose put
        - continue to t2

    At t2 the holder can:
        - choose call
        - choose put

    Parameters
    ----------
    values_func
        merton_values or kou_values

    seed
        Optional random seed.

    Returns
    -------
    price
    stderr
    ci95
    """

    if seed is not None:
        np.random.seed(seed)

    #
    # Outer simulation to t1
    #

    S_t1 = values_func(
        S0=S0,
        r=r,
        times=[t1],
        n_paths=n_outer,
        **params,
    )[:, 0]

    outer_values = np.empty(n_outer)

    disc_1T = np.exp(-r * (T - t1))
    disc_12 = np.exp(-r * (t2 - t1))
    disc_01 = np.exp(-r * t1)
    disc_2T = np.exp(-r * (T - t2))

    for i, s1 in enumerate(S_t1):

        #
        # Choose immediately at t1
        #

        ST = values_func(
            S0=s1,
            r=r,
            times=[T - t1],
            n_paths=n_inner_t1,
            **params,
        )[:, -1]

        call_value_t1 = disc_1T * np.mean(
            call(ST, Kc)
        )

        put_value_t1 = disc_1T * np.mean(
            put(ST, Kp)
        )

        #
        # Continuation value
        #

        S_t2 = values_func(
            S0=s1,
            r=r,
            times=[t2 - t1],
            n_paths=n_inner_t1,
            **params,
        )[:, 0]

        values_t2 = np.empty(n_inner_t1)

        for j, s2 in enumerate(S_t2):

            ST2 = values_func(
                S0=s2,
                r=r,
                times=[T - t2],
                n_paths=n_inner_t2,
                **params,
            )[:, -1]

            call_value_t2 = disc_2T * np.mean(
                call(ST2, Kc)
            )

            put_value_t2 = disc_2T * np.mean(
                put(ST2, Kp)
            )

            values_t2[j] = max(
                call_value_t2,
                put_value_t2,
            )

        continuation_value = (
            disc_12 * np.mean(values_t2)
        )

        outer_values[i] = max(
            call_value_t1,
            put_value_t1,
            continuation_value,
        )

    #
    # Price and confidence interval
    #

    price = disc_01 * np.mean(
        outer_values
    )

    stderr = (
        disc_01
        * np.std(outer_values, ddof=1)
        / np.sqrt(n_outer)
    )

    ci95 = (
        price - 1.96 * stderr,
        price + 1.96 * stderr,
    )

    return price, stderr, ci95
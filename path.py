import numpy as np

def merton_values(
    S0,
    r,
    sigma,
    xiP,
    muJ,
    sigmaJ,
    times: list, # List of times. The last is maturity and the previous are choice times, example: [t1,t2,tau]
    n_paths,
    q=0.0,
):
    omega = xiP * (
        np.exp(muJ + 0.5 * sigmaJ**2) - 1.0
    )

    mu = r - q - 0.5 * sigma**2 - omega

    times = np.asarray(times)

    values = np.empty((n_paths, len(times)))

    X = np.zeros(n_paths)

    previous_time = 0.0

    for j, t in enumerate(times):

        dt = t - previous_time

        N = np.random.poisson(
            xiP * dt,
            size=n_paths,
        )

        Z = np.random.randn(n_paths)
        ZJ = np.random.randn(n_paths)

        X += (
            mu * dt
            + sigma * np.sqrt(dt) * Z
            + N * muJ
            + sigmaJ * np.sqrt(N) * ZJ
        )

        values[:, j] = S0 * np.exp(X)

        previous_time = t

    return values

def kou_values(
    S0,
    r,
    sigma,
    xiP,
    p1,
    alpha1,
    alpha2,
    times,
    n_paths,
    q=0.0,
):
    omega = xiP * (
        p1 * alpha1 / (alpha1 - 1.0)
        + (1.0 - p1) * alpha2 / (alpha2 + 1.0)
        - 1.0
    )

    mu = r - q - 0.5 * sigma**2 - omega

    times = np.asarray(times)

    values = np.empty((n_paths, len(times)))

    X = np.zeros(n_paths)

    previous_time = 0.0

    for j, t in enumerate(times):

        dt = t - previous_time

        Z = np.random.randn(n_paths)

        dX = (
            mu * dt
            + sigma * np.sqrt(dt) * Z
        )

        N = np.random.poisson(
            xiP * dt,
            size=n_paths,
        )

        jump_sum = np.zeros(n_paths)

        for i in range(n_paths):

            n = N[i]

            if n > 0:

                U = np.random.rand(n)

                jumps = np.where(
                    U < p1,
                    np.random.exponential(
                        1.0 / alpha1,
                        size=n,
                    ),
                    -np.random.exponential(
                        1.0 / alpha2,
                        size=n,
                    ),
                )

                jump_sum[i] = jumps.sum()

        X += dX + jump_sum

        values[:, j] = S0 * np.exp(X)

        previous_time = t

    return values
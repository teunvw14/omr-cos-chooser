import numpy as np


def gbm_generate_paths(S_0, mu, sigma, tau, n_paths, n_steps, rng):
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = np.log(S_0)

    dt = tau / n_steps

    for i in range(n_steps):
        X[:, i+1] = X[:, i] + (mu - 0.5*sigma*sigma) * dt + sigma * np.sqrt(dt) * rng.normal(size=n_paths)

    return np.exp(X)

def merton_generate_paths(S_0, mu, sigma, jump_rate, jump_mean, jump_std, tau, n_paths, n_steps, rng):
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = np.log(S_0)

    dt = tau / n_steps
    for i in range(n_steps):
        jumps = rng.poisson(dt * jump_rate, size=n_paths)
        X[:, i+1] = X[:, i] + mu * dt + sigma * np.sqrt(dt) * rng.normal(size=n_paths) + jumps * (jump_mean + jump_std * rng.normal(size=n_paths))

    return np.exp(X)

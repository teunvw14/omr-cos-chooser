import numpy as np


def call(ST, K):
    return np.maximum(ST - K, 0.0)


def put(ST, K):
    return np.maximum(K - ST, 0.0)


def chooser(ST, Kc, Kp=None):
    if Kp is None:
        Kp = Kc
    return np.maximum(call(ST, Kc), put(ST, Kp))
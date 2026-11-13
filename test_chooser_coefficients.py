import numpy as np
from cos import chooser_coeffs
from integrals import frequencies

N = 16

A_call = np.random.rand(N)
B_call = np.random.rand(N)

A_put = np.random.rand(N)
B_put = np.random.rand(N)

omega = frequencies(N, -5, 5)
nu = frequencies(N, -5, 5)

H = chooser_coeffs(
    A_call,
    B_call,
    A_put,
    B_put,
    omega,
    nu,
    -5.0,
    -5.0,
    5.0,
    0.0,
)

print(H.shape)
print(np.all(np.isfinite(H)))
import numpy as np
from coefficients import backward_step

A = np.random.rand(16)
B = np.random.rand(16)

Ic = np.random.rand(16,16)
Is = np.random.rand(16,16)
Io = np.random.rand(16)

H = backward_step(
    A,
    B,
    100.0,
    Ic,
    Is,
    Io,
)

print(H.shape)
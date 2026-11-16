# test_integrals.py

import numpy as np

from integrals import (
    build_Io,
    build_Ic,
    build_Is,
)

a = 3.205956623614996
b = 6.034383748361186

ap = 1.8067430612419009
bp = 7.463597310734281

xstar = 4.580170185987937

Nin = 32
Nout = 32

omega = np.arange(Nin) * np.pi / (bp - ap)
nu    = np.arange(Nout) * np.pi / (b - a)

Io = build_Io(
    nu,
    a,
    b,
    xstar,
    b,
)

Ic = build_Ic(
    omega,
    nu,
    a,
    ap,
    b,
    xstar,
    b,
)

Is = build_Is(
    omega,
    nu,
    a,
    ap,
    b,
    xstar,
    b,
)

print()
print("Io")

print("shape =", Io.shape)
print("nan   =", np.isnan(Io).sum())
print("inf   =", np.isinf(Io).sum())
print("max   =", np.max(np.abs(Io)))

print()
print("Ic")

print("shape =", Ic.shape)
print("nan   =", np.isnan(Ic).sum())
print("inf   =", np.isinf(Ic).sum())
print("max   =", np.max(np.abs(Ic)))

print()
print("Is")

print("shape =", Is.shape)
print("nan   =", np.isnan(Is).sum())
print("inf   =", np.isinf(Is).sum())
print("max   =", np.max(np.abs(Is)))

print()
print("Diagonal Ic")

for i in range(10):
    print(i, Ic[i, i])

print()
print("Diagonal Is")

for i in range(10):
    print(i, Is[i, i])
import sympy as sp

# symbols
x = sp.symbols("x", real=True)

a1, a2 = sp.symbols("a1 a2", real=True)
b = sp.symbols("b", real=True)
x_star = sp.symbols("x_star", real=True)

omega_k = sp.symbols("omega_k", real=True)
omega_n = sp.symbols("omega_n", real=True)

#
# Io
#

Io = (
    2 / (b - a1)
    * sp.integrate(
        sp.cos(
            omega_n * (x - a1)
        ),
        (x, x_star, b),
    )
)

Io = sp.simplify(Io)

#
# Ic
#

Ic = (
    2 / (b - a1)
    * sp.integrate(
        sp.cos(
            omega_k * (x - a2)
        )
        * sp.cos(
            omega_n * (x - a1)
        ),
        (x, x_star, b),
    )
)

Ic = sp.trigsimp(
    sp.simplify(Ic)
)

#
# Is
#

Is = (
    2 / (b - a1)
    * sp.integrate(
        sp.sin(
            omega_k * (x - a2)
        )
        * sp.cos(
            omega_n * (x - a1)
        ),
        (x, x_star, b),
    )
)

Is = sp.trigsimp(
    sp.simplify(Is)
)

print("\nIo:\n")
sp.pprint(Io)

print("\nIc:\n")
sp.pprint(Ic)

print("\nIs:\n")
sp.pprint(Is)
print("\n")
print(sp.pycode(Io))
print("\n")
print(sp.pycode(Ic))
print("\n")
print(sp.pycode(Is))
print("\n")
print(sp.latex(Io))
print("\n")
print(sp.latex(Ic))
print("\n")
print(sp.latex(Is))
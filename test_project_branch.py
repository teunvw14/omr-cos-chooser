import numpy as np

from cos import (
    branch_values,
    find_winning_regions,
)


# --------------------------------------------------
# Common interval
# --------------------------------------------------

a = 3.0
b = 6.0

N = 32


# --------------------------------------------------
# Artificial branch coefficients
#
# These produce three branches with the intended
# ordering:
#
#     put -> wait -> call
#
# across the interval [a, b].
# --------------------------------------------------

A_put = np.zeros(N)
B_put = np.zeros(N)

A_put[0] = 12.0
A_put[1] = 8.0


A_wait = np.zeros(N)
B_wait = np.zeros(N)

A_wait[0] = 18.0
A_wait[2] = -4.0


A_call = np.zeros(N)
B_call = np.zeros(N)

A_call[0] = 12.0
A_call[1] = -8.0


# --------------------------------------------------
# Branch functions
# --------------------------------------------------

def put_function(x):
    return branch_values(
        x=x,
        A=A_put,
        B=B_put,
        a_source=a,
        b_source=b,
    )


def wait_function(x):
    return branch_values(
        x=x,
        A=A_wait,
        B=B_wait,
        a_source=a,
        b_source=b,
    )


def call_function(x):
    return branch_values(
        x=x,
        A=A_call,
        B=B_call,
        a_source=a,
        b_source=b,
    )


# --------------------------------------------------
# Find the winning regions
# --------------------------------------------------

regions = find_winning_regions(
    value_functions=[
        put_function,
        wait_function,
        call_function,
    ],
    branch_names=[
        "put",
        "wait",
        "call",
    ],
    a=a,
    b=b,
    grid_size=2001,
    root_tol=1e-12,
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print("Winning regions")

for left, right, branch_index, branch_name in regions:
    print(
        f"[{left:.12f}, {right:.12f}] "
        f"-> {branch_name}"
    )


# --------------------------------------------------
# Basic validation
# --------------------------------------------------

expected_names = [
    "put",
    "wait",
    "call",
]

actual_names = [
    region[3]
    for region in regions
]

if actual_names != expected_names:
    raise AssertionError(
        "Unexpected branch ordering.\n"
        f"Expected: {expected_names}\n"
        f"Actual:   {actual_names}"
    )

if len(regions) != 3:
    raise AssertionError(
        f"Expected 3 regions, found {len(regions)}."
    )

for left, right, _, name in regions:
    if not left < right:
        raise AssertionError(
            f"Invalid interval for branch {name}: "
            f"[{left}, {right}]"
        )


# Check continuity at the two detected boundaries.

first_boundary = regions[0][1]
second_boundary = regions[1][1]

put_wait_error = abs(
    put_function(
        np.array([first_boundary])
    )[0]
    -
    wait_function(
        np.array([first_boundary])
    )[0]
)

wait_call_error = abs(
    wait_function(
        np.array([second_boundary])
    )[0]
    -
    call_function(
        np.array([second_boundary])
    )[0]
)

print()
print(
    "Put/wait boundary error :",
    f"{put_wait_error:.6e}",
)

print(
    "Wait/call boundary error:",
    f"{wait_call_error:.6e}",
)

tolerance = 1e-10

if put_wait_error > tolerance:
    raise AssertionError(
        "Put and wait values do not match "
        "at the first boundary."
    )

if wait_call_error > tolerance:
    raise AssertionError(
        "Wait and call values do not match "
        "at the second boundary."
    )

print()
print("Three-branch region test passed.")
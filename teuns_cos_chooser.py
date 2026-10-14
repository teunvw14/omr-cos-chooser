
def chooser_inner_integral(n, K, r, T, tc, a, b, N):
    result = 0

    A_0 = 
    B_0 = 

def chooser_outer_cosine_coeff(n, K, r, T, tc, a, b, N):
    x_1 = np.log(K) - r * (T - tc)
    nu_n = n * np.pi / (b-a)

    I_1 = 2 * K * np.exp(-r * (T-tc)) * np.sin(nu_n * (x_1-a)) / ((b-a) * nu_n)
    I_2 = (2 * np.exp(x_1) * (np.cos(nu_n * (x_1-a) + nu_n * np.sin(nu_n * (x_1-a))))) / ((b-a) * (nu_n**2 + 1))

    if n == 0:
        I_1 = 2 * K * np.exp(-r * (T-tc)) * (x_1-a) / (b-a)
        I_2 = 2 * (np.exp(x_1) - np.exp(a)) / (b-a)


def price_chooser_cos(cf, S0, K, r, tau, tc, N, L, a, b):
    value = 0
    x_0 = np.log(S0)

    nu_0 = 0
    H_0 = chooser_outer_cosine_coeff(0)
    value += 0.5 * np.real(cf(nu_0) * np.exp(1j * nu_0 * (x_0 - a))) * H_0

    for n in range(N):
        nu_n = n * np.pi / (b-a)
        H_k = chooser_outer_cosine_coeff(n)

    discount = np.exp(-r * tau)
    return discount * value 

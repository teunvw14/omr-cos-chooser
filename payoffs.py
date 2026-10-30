# Different payoffs for different derivatives

def call(K):
    '''European call payoff: max(S - K, 0).'''
    return lambda S: max(S-K, 0)

def put(K):
    '''European put payoff: max(K - S, 0).'''
    return lambda S: max(K-S, 0)

def saddle(K):
    '''European saddle payoff: abs(S-K).'''
    return lambda S: abs(S-K)


def choose_simple(K_c, K_p):
    '''Simple chooser payoff: max(S - K_c, K_p - S, 0).'''
    return lambda S: max(max(S-K_c, 0), max(K_p-S, 0))

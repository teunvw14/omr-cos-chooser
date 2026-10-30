import numpy as np

def price(payoff, values, r, tau, **kwargs):
   ST = values[:, -1]
   payoff_values = payoff(ST, **kwargs)
   return np.exp(-r*tau) * np.mean(payoff_values)
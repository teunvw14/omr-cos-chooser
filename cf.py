import numpy as np

def gbm(r, sigma, q=0):
    mu = r - 0.5*sigma**2 - q

    def phi(u, tau):
        return np.exp(1j*u*mu*tau - 0.5*sigma**2*u**2*tau)

    def cumulants(tau):
        c1 = mu * tau
        c2 = sigma**2 * tau
        c4 = 0.0
        return c1, c2, c4

    return phi, cumulants

def merton(r, sigma, xiP, muJ, sigmaJ, q=0):
    omega = xiP * (np.exp(0.5*sigmaJ**2 + muJ) - 1.0)
    mu    = r - 0.5*sigma**2 - q - omega

    def phi(u, tau):
        return np.exp(
            1j*u*mu*tau - 0.5*sigma**2*u**2*tau
            + xiP*tau*(np.exp(1j*muJ*u - 0.5*sigmaJ**2*u**2)- 1.0))

    def cumulants(tau):
        c1 = tau * (mu + xiP*muJ)
        c2 = tau * (sigma**2 + xiP*(muJ**2 + sigmaJ**2))
        c4 = tau * xiP * (muJ**4 + 6*sigmaJ**2*muJ**2 + 3*sigmaJ**4)
        return c1, c2, c4
    
    return phi, cumulants


def kou(r, sigma, xiP, p1, alpha1, alpha2, q=0):
    omega = xiP*(1.0 - p1*alpha1/(alpha1 - 1.0)
          - (1.0-p1)*alpha2/(alpha2 + 1.0)) # 1-p1 is p2
    mu    = r - 0.5*sigma**2 - q + omega   

    def phi(u, tau):
        return np.exp(
            1j*u*mu*tau - 0.5*sigma**2*u**2*tau
            + xiP*tau*(p1*alpha1/(alpha1 - 1j*u)
                       + (1.0-p1)*alpha2/(alpha2 + 1j*u) - 1.0))

    def cumulants(tau):
        c1 = tau * (mu + xiP*(p1/alpha1 - (1.0-p1)/alpha2))
        c2 = tau * (sigma**2 + 2*xiP*(p1/alpha1**2 + (1.0-p1)/alpha2**2))
        c4 = tau * 24*xiP*(p1/alpha1**4 + (1.0-p1)/alpha2**4)
        return c1, c2, c4
    
    return phi, cumulants

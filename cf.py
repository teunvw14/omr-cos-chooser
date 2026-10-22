import numpy as np

def GBM(r, sigma, q=0):
    mu = r - 0.5*sigma**2 - q

    def phi(u, tau):
        return np.exp(1j*u*mu*tau - 0.5*sigma**2*u**2*tau)

    return phi

def Merton(r, sigma, xiP, muJ, sigmaJ, q=0):
    omega = xiP * (np.exp(0.5*sigmaJ**2 + muJ) - 1.0)
    mu    = r - 0.5*sigma**2 - q - omega

    def phi(u, tau):
        return np.exp(
            1j*u*mu*tau - 0.5*sigma**2*u**2*tau
            + xiP*tau*(np.exp(1j*muJ*u - 0.5*sigmaJ**2*u**2)- 1.0))

    return phi

def Kou(r, sigma, xiP, p1, alpha1, alpha2, q=0):
    omega = xiP*(1.0 - p1*alpha1/(alpha1 - 1.0)
          - (1.0-p1)*alpha2/(alpha2 + 1.0)) # 1-p1 is p2
    mu    = r - 0.5*sigma**2 - q + omega   

    def phi(u, tau):
        return np.exp(
            1j*u*mu*tau - 0.5*sigma**2*u**2*tau
            + xiP*tau*(p1*alpha1/(alpha1 - 1j*u)
                       + (1.0-p1)*alpha2/(alpha2 + 1j*u) - 1.0))

    return phi

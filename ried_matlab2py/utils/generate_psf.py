import numpy as np

from scipy.integrate import quad
from scipy.special import j0

def radial_amplitude(r, lmda, NA, z):

    u = 4 * np.pi * z * NA**2 / lmda

    def integrand(p):
        phase = np.exp(0.5j * u * p**2)
        bessel = j0(2 * np.pi * r * NA / lmda * p)
        return 2 * phase * bessel

    real = quad(lambda p: np.real(integrand(p)), 0, 1)[0]
    imag = quad(lambda p: np.imag(integrand(p)), 0, 1)[0]

    return real + 1j * imag

def generate_psf(pixel, lmda, n, NA, z):

    #h=@(r,p) 2*exp((1i*u*(p.^2))/2).*besselj(0,2*pi*r*NA/lamda.*p); is removed because it is not used in the function in Python
    x = np.arange(-n * pixel, (n + 1) * pixel, pixel)
    X, Y = np.meshgrid(x, x)

    r = np.sqrt(X**2 + Y**2)

    mask = r <= 1

    ipsf = np.zeros_like(r, dtype=np.complex128)
    
    for i, j in np.argwhere(mask):
        ipsf[i, j] = radial_amplitude(r[i, j], lmda, NA, z)

    ipsf = np.abs(ipsf)**2
    ipsf /= np.sum(ipsf)

    return ipsf

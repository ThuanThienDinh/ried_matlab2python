from .generate_psf import generate_psf
import numpy as np

def kernel(pixel, lmda, n, NA, z):

    if z is None:
        z = 0
    if n is None:
        n = 17

    if 17 < n < 33:
        nn = 8
    elif 33 <= n < 65:
        nn = 16
    elif 65 <= n < 129:
        nn = 32
    elif 129 <= n < 257:
        nn = 64
    else:
        nn = 64

    psf = generate_psf(pixel, lmda, nn, NA, z)
    psf = psf / np.sum(psf)

    return psf 
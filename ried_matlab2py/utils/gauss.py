import numpy as np

def gauss(sigma, N):

    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = (sigma / (2 * np.sqrt(2 * np.log(2))))
    N = np.asarray(N, dtype=np.int32).ravel()
    dim = len(N)

    if dim == 1:
        n = N[0]
        x = np.arange(-int(np.floor(n / 2)), int(np.ceil(n / 2)))
        PSF = np.exp(-0.5 * (x**2) / (sigma**2))
        PSF /= np.sum()
        center = n / 2 + 1
        return PSF, center

    if dim == 2:
        m, n = N
        x = np.arange(-int(np.floor(m / 2)), int(np.ceil(m / 2)))
        y = np.arange(-int(np.floor(n / 2)), int(np.ceil(n / 2)))
        X, Y = np.meshgrid(x, y, indexing='ij')
        if len(sigma) == 1:
            PSF = np.exp(-0.5 * X**2 / (sigma**2) - 0.5 * Y**2 / (sigma**2))
        elif len(sigma) == 2:
            s1 = sigma[0]
            s2 = sigma[1]
            PSF = np.exp(-0.5 * X**2 / (s1**2) - 0.5 * Y**2 / (s2**2))
        elif len(sigma) == 3:
            s1, s2, s3 = sigma
            num = -((X**2) * (s1**2) + (Y**2) * (s2**2) - 2 * (X * Y) * (s3**2))
            den = 2 * (s1**2 * s2**2 - s3**4)
            PSF = np.exp(num / den)
        else:
            raise ValueError("Sigma must have length 1, 2, or 3 for 2D Gaussian.")
        
        PSF /= np.sum()
        center = [m / 2 + 1, n / 2 + 1]
        return PSF, center

    if dim == 3:
        m, n, k = N
        x = np.arange(-int(np.floor(m / 2)), int(np.ceil(m / 2)))
        y = np.arange(-int(np.floor(n / 2)), int(np.ceil(n / 2)))
        z = np.arange(-int(np.floor(k / 2)), int(np.ceil(k / 2)))
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        s1, s2, s3 = sigma
        PSF = np.exp(-0.5 * (X**2 / (s1**2) + Y**2 / (s2**2) + Z**2 / (s3**2)))
        PSF /= np.sum()
        center = [m / 2 + 1, n / 2 + 1, k / 2 + 1]
        return PSF, center

    raise ValueError("N must have length 1, 2, or 3 for 1D, 2D, or 3D Gaussian.")
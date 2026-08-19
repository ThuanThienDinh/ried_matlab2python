import math
import abc
import numpy as np

class scipy(abc.ABC):
    """Lightweight compatibility layer for the subset of scipy used here."""

    @staticmethod
    @abc.abstractmethod
    def erf(x):
        raise NotImplementedError("scipy.erf must be implemented.")

    @staticmethod
    @abc.abstractmethod
    def erfc(x):
        raise NotImplementedError("scipy.erfc must be implemented.")

    class special:
        @staticmethod
        def erf(x):
            x = np.asarray(x, dtype=np.float64)
            return np.vectorize(math.erf)(x)

        @staticmethod
        def erfc(x):
            x = np.asarray(x, dtype=np.float64)
            return np.vectorize(lambda z: 1.0 - math.erf(z))(x)

def generate_rsf(gama, n):

    if n is None:
        n = int(np.ceil(gama * np.sqrt(-2 * np.log(0.0002))/ np.sqrt(8 * np.log(2)))) + 1

    sigma = gama / np.sqrt(8 * np.log(2))
    kernel_radius = min(int(np.ceil(sigma * np.sqrt(-2 * np.log(0.0002))) + 1), int(np.floor(n / 2)))
    ii = np.arange(-kernel_radius, kernel_radius + 1)
    rsf_x = 0.5 * (scipy.special.erf((ii + 0.5) / (sigma * np.sqrt(2))) - scipy.special.erf((ii - 0.5) / (sigma * np.sqrt(2))))
    kernel = np.outer(rsf_x, rsf_x)

    return kernel / np.sum(kernel)





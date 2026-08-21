import numpy as np

def calprob(data, bin_size, minv, maxv):
    data = np.asarray(data, dtype=np.float64).ravel()
    bin_size = int(bin_size)
    if bin_size < 1:
        raise ValueError("bin_size must be positive.")
    if maxv == minv:
        output = np.zeros(bin_size, dtype=np.float64)
        output[0] = 1.0
        return output

    indices = np.rint((data - minv) / (maxv - minv) * bin_size).astype(np.int64)
    indices = np.clip(indices, 0, bin_size - 1)
    output = np.bincount(indices, minlength=bin_size).astype(np.float64)
    output /= output.sum()
    return output
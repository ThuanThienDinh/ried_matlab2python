import numpy as np

def calprob(data, bin_size, minv, maxv):

    data = np.asarray(data).ravel()

    output = np.zeros(int(bin_size), dtype=np.float64)

    for value in data:
        idx = int(np.round((value - minv) / (maxv - minv) * (bin_size)))
        if idx < 0:
            idx = 0
        if idx >= bin_size:
            idx = bin_size - 1

        output[idx] += 1

    output /= np.sum(output)
    return output
import numpy as np

def binning(indata, bin_size):

    rows, cols = indata.shape

    if rows % bin_size != 0 or cols % bin_size != 0:
        raise ValueError("Data dimensions must be divisible by bin size.")

    output = indata.reshape(rows // bin_size, bin_size, cols // bin_size, bin_size).mean(axis=(1, 3))

    return output
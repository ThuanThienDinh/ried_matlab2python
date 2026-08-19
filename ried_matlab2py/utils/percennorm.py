import numpy as np

def percennorm(data, minper = 0, maxper = 100):

    data = np.array(data, dtype=np.float64)

    data_min = np.percentile(data, minper)
    data_max = np.percentile(data, maxper)
    output = (data - data_min) / (data_max - data_min)
    output = np.clip(output, 0, 1)

    return output

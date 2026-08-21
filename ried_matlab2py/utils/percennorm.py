import numpy as np

def percennorm(data, minper = 0, maxper = 100):

    data = np.array(data, dtype=np.float64)

    data_min = np.percentile(data, minper)
    data_max = np.percentile(data, maxper)
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        raise ValueError("data must contain finite values for percentile normalization")
    if data_max <= data_min:
        return np.zeros_like(data, dtype=np.float64)
    output = (data - data_min) / (data_max - data_min)
    output = np.clip(output, 0, 1)

    return output

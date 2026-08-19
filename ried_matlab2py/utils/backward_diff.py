import numpy as np

def backward_diff(data, step, dim):

    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D array, got {data.ndim}D array instead.")
    if dim not in [0, 1, 2]:
        raise ValueError("Dimension must be 0, 1, or 2, got {dim} instead.")
    if step == 0:
        raise ValueError("Step size must be non-zero.")

    axis = dim

    out = np.empty_like(data, dtype=np.result_type(data, np.float64))

    current = [slice(None)] * 3
    previous = [slice(None)] * 3

    current[axis] = slice(1, None)
    previous[axis] = slice(0, -1)

    out[tuple(current)] = (data[tuple(current)] - data[tuple(previous)]) / step

    first = [slice(None)] * 3
    first[axis] = 0

    out[tuple(first)] = (data[tuple(first)]) / step

    return out
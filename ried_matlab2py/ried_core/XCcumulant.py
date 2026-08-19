"""Cross-cumulant computation for 3D volumetric data."""

import numpy as np


def XCcumulant(data, order, offset):

    data = np.asarray(data, dtype=np.float64)
    
    # Get spatial dimensions
    x_size, y_size = data.shape[0], data.shape[1]
    
    # Initialize output array with doubled spatial dimensions
    # MATLAB: zeros(size(data,1)*2, size(data,2)*2)
    output = np.zeros((x_size * 2, y_size * 2), dtype=np.float64)
    
    # Compute spectral mean along the third dimension
    # MATLAB: mean(data, 3)
    temporal_mean = np.mean(data, axis=2, keepdims=True)
    
    # Center and power the data
    # MATLAB: cdata = abs(data - offset * mean(data,3)).^order
    cdata = np.abs(data - offset * temporal_mean) ** order
    
    # Fill output[2:2:end-1, 1:2:end] = mean(cdata(1:end-1,:,:) * cdata(2:end,:,:), 3)
    # MATLAB row 2:2:end-1 (3 rows: 2,4,6) → Python row 1:2*x-1:2
    # MATLAB col 1:2:end (5 cols: 1,3,5,7,9) → Python col 0:2*y:2
    output[1:(2*x_size-1):2, 0:(2*y_size):2] = np.mean(
        cdata[:-1, :, :] * cdata[1:, :, :], axis=2
    )
    
    # Fill output[1:2:end, 2:2:end-1] = mean(cdata(:,1:end-1,:) * cdata(:,2:end,:), 3)
    # MATLAB row 1:2:end (4 rows: 1,3,5,7) → Python row 0:2*x_size:2
    # MATLAB col 2:2:end-1 (4 cols: 2,4,6,8) → Python col 1:2*y_size-1:2
    output[0:(2*x_size):2, 1:(2*y_size-1):2] = np.mean(
        cdata[:, :-1, :] * cdata[:, 1:, :], axis=2
    )
    
    # Fill output[2:2:end-1, 2:2:end-1] with diagonal average
    # MATLAB row 2:2:end-1 (3 rows: 2,4,6) → Python row 1:2*x-1:2
    # MATLAB col 2:2:end-1 (4 cols: 2,4,6,8) → Python col 1:2*y-1:2
    # Average of two diagonal cross-products
    diag1 = np.mean(
        cdata[:-1, :-1, :] * cdata[1:, 1:, :], axis=2
    )
    diag2 = np.mean(
        cdata[:-1, 1:, :] * cdata[1:, :-1, :], axis=2
    )
    output[1:(2*x_size-1):2, 1:(2*y_size-1):2] = (diag1 + diag2) / 2
    
    # Fill output[1:2:end, 1:2:end] = mean(cdata(:,:,1:end-1) * cdata(:,:,2:end), 3)
    # MATLAB row 1:2:end (4 rows: 1,3,5,7) → Python row 0:2*x_size:2
    # MATLAB col 1:2:end (5 cols: 1,3,5,7,9) → Python col 0:2*y_size:2
    # This computes temporal neighbors (along z dimension)
    output[0:(2*x_size):2, 0:(2*y_size):2] = np.mean(
        cdata[:, :, :-1] * cdata[:, :, 1:], axis=2
    )
    return output
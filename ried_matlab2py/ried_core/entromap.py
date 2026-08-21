"""Entropy map computation for 3D volumetric data."""

import sys
from pathlib import Path
import numpy as np

# Handle both direct execution and package import
try:
    from .xEtr import xEtr
except ImportError:
    try:
        from xEtr import xEtr
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from xEtr import xEtr


def entromap(data, bin, maxv, minv):
    """
    Compute entropy map from 3D volumetric data.
    
    This function creates an entropy map by computing cross-entropy measures
    between neighboring voxels in 3D data. The output has dimensions (2*x, 2*y)
    where each 2x2 block corresponds to a neighborhood relationship.
    
    Parameters
    ----------
    data : np.ndarray
        3D input data array of shape (x, y, z).
    bin : int
        Number of bins for probability distribution calculation.
    maxv : float
        Maximum value for binning range.
    minv : float
        Minimum value for binning range.
    
    Returns
    -------
    np.ndarray
        Entropy map of shape (2*x, 2*y) containing:
        - (2*i-1, 2*j-1): Self-entropy of data[i,j,:]
        - (2*i, 2*j-1) and (2*i-1, 2*j): Cross-entropy with neighbors
        - (2*i, 2*j): Average cross-entropy of opposite corners
    
    Notes
    -----
    The entropy map encodes local similarity patterns in the volume:
    - Diagonal (m+n==0): Self-entropy (always zero cross-entropy)
    - Edge-adjacent (m+n==1): Cross-entropy with horizontal/vertical neighbors
    - Diagonal-adjacent (m+n==2): Average cross-entropy with diagonal neighbors
    """
    data = np.asarray(data, dtype=np.float64)
    
    # Get dimensions (MATLAB: [x,y,~] = size(data))
    x, y = data.shape[0], data.shape[1]
    
    # Initialize output array (MATLAB: zeros(x*2, y*2))
    output = np.zeros((x * 2, y * 2), dtype=np.float64)
    
    flat = data.reshape(x * y, data.shape[2])
    if maxv == minv:
        probabilities = np.zeros((x * y, int(bin)), dtype=np.float64)
        probabilities[:, 0] = 1.0
    else:
        indices = np.rint((flat - minv) / (maxv - minv) * int(bin)).astype(np.int64)
        indices = np.clip(indices, 0, int(bin) - 1)
        probabilities = np.zeros((x * y, int(bin)), dtype=np.float64)
        rows = np.repeat(np.arange(x * y), data.shape[2])
        np.add.at(probabilities, (rows, indices.ravel()), 1)
        probabilities /= data.shape[2]

    def cross_entropy(first, second):
        first_term = np.zeros_like(first)
        second_term = np.zeros_like(second)
        first_mask = first > 0
        second_mask = second > 0
        first_term[first_mask] = second[first_mask] * np.log(first[first_mask])
        second_term[second_mask] = first[second_mask] * np.log(second[second_mask])
        return -0.5 * np.sum(first_term + second_term, axis=-1)

    probabilities = probabilities.reshape(x, y, int(bin))
    center = probabilities[:-1, :-1]
    output[0:2 * (x - 1):2, 1:2 * (y - 1):2] = cross_entropy(
        center, probabilities[:-1, 1:]
    )
    output[1:2 * (x - 1):2, 0:2 * (y - 1):2] = cross_entropy(
        center, probabilities[1:, :-1]
    )
    output[0:2 * (x - 1):2, 0:2 * (y - 1):2] = cross_entropy(
        center, center
    )
    output[1:2 * (x - 1):2, 1:2 * (y - 1):2] = (
        cross_entropy(center, probabilities[1:, 1:])
        + cross_entropy(probabilities[:-1, 1:], probabilities[1:, :-1])
    ) / 2
    
    return output

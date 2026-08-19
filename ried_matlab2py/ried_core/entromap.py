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
    
    # Nested loops over spatial coordinates
    # MATLAB: for i = 1:x-1, for j = 1:y-1
    # Python: for i in range(x-1), for j in range(y-1) with index adjustment
    for i in range(x - 1):
        for j in range(y - 1):
            # Inner loops for neighborhood (MATLAB: for m = 0:1, for n = 0:1)
            for m in range(2):
                for n in range(2):
                    
                    # Case 1: Self-entropy (m+n == 0)
                    if m + n == 0:
                        # MATLAB: p1 = squeeze(data(i,j,:))
                        p1 = np.squeeze(data[i, j, :])
                        # MATLAB: output(2*i-1, 2*j-1) = xEtr(p1,p1,bin,maxv,minv)
                        # Convert MATLAB 1-based to Python 0-based indexing
                        output[2 * i, 2 * j] = xEtr(p1, p1, bin, maxv, minv)
                    
                    # Case 2: Adjacent neighbor (m+n == 1)
                    if m + n == 1:
                        p1 = np.squeeze(data[i, j, :])
                        # MATLAB: p2 = squeeze(data(i+m,j+n,:))
                        p2 = np.squeeze(data[i + m, j + n, :])
                        # MATLAB: output(2*i-1+m, 2*j-1+n) = xEtr(p1,p2,bin,maxv,minv)
                        output[2 * i + m, 2 * j + n] = xEtr(p1, p2, bin, maxv, minv)
                    
                    # Case 3: Opposite corners (m+n == 2)
                    if m + n == 2:
                        p1 = np.squeeze(data[i, j, :])
                        # MATLAB: p2 = squeeze(data(i+m,j,:))
                        p2 = np.squeeze(data[i + m, j, :])
                        # MATLAB: p3 = squeeze(data(i,j+n,:))
                        p3 = np.squeeze(data[i, j + n, :])
                        # MATLAB: p4 = squeeze(data(i+m,j+n,:))
                        p4 = np.squeeze(data[i + m, j + n, :])
                        # MATLAB: output(2*i+m-1, 2*j+n-1) = ((xEtr(p1,p4,...)+xEtr(p2,p3,...))/2)
                        output[2 * i + m - 1, 2 * j + n - 1] = (
                            (xEtr(p1, p4, bin, maxv, minv) + xEtr(p2, p3, bin, maxv, minv)) / 2
                        )
    
    return output

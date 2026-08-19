"""Cross-entropy metric between two probability distributions."""

import sys
from pathlib import Path
import numpy as np

try:
    from utils.calprob import calprob
except ImportError:
    try:
        from utils.calprob import calprob
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.calprob import calprob

from utils.calprob import calprob

def xEtr(data1, data2, bin, maxv, minv):
    """
    Compute symmetric cross-entropy between two data distributions.
    
    This function calculates a symmetric divergence measure between two datasets
    by computing their probability distributions and calculating the cross-entropy.
    
    Parameters
    ----------
    data1 : array-like
        First input data array.
    data2 : array-like
        Second input data array.
    bin : int
        Number of bins for probability distribution calculation.
    maxv : float
        Maximum value for binning range.
    minv : float
        Minimum value for binning range.
    
    Returns
    -------
    float
        Symmetric cross-entropy value. Smaller values indicate higher similarity.
    
    Notes
    -----
    The metric is calculated as:
        xEtr = -sum(P2(i)*log(P1(i)) + P1(i)*log(P2(i))) / 2
    
    where P1 and P2 are the probability distributions of data1 and data2.
    """
    # Compute probability distributions for both datasets
    P1 = calprob(data1, bin, minv, maxv)
    P2 = calprob(data2, bin, minv, maxv)
    
    # Initialize sum accumulator
    h_sum = 0
    
    # Compute symmetric cross-entropy
    for i in range(len(P1)):
        if P1[i] > 0:
            h_sum += P2[i] * np.log(P1[i])
        if P2[i] > 0:
            h_sum += P1[i] * np.log(P2[i])
    
    # Return negative normalized sum
    output = -h_sum / 2
    
    return output

if __name__ == "__main__":
    print("xEtr module loaded successfully")
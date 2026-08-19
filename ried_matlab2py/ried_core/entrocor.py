import numpy as np
import sys
from pathlib import Path

# Handle both direct execution and package import
try:
    from .entromap import entromap
    from .XCcumulant import XCcumulant
except ImportError:
    try:
        from entromap import entromap
        from XCcumulant import XCcumulant
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from entromap import entromap
        from XCcumulant import XCcumulant

def entrocor(data, bin, offset, maxv, minv):

    entrom = entromap(data, bin, maxv, minv)
    ecum = XCcumulant(data, 1, offset)
    output = entrom * ecum
    output /= np.max(output)

    return(output)
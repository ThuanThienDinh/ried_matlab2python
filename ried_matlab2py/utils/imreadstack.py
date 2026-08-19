from pathlib import Path

import importlib

def imreadstack(imname):
    try:
        tifffile = importlib.import_module("tifffile")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tifffile is required to read TIFF stacks. Install it with 'pip install tifffile'."
        ) from exc

    return tifffile.imread(imname)

import importlib
from pathlib import Path

import numpy as np

def imreadstack(imname):
    """Read an ImageJ/multipage TIFF as ``(frames, rows, columns)``."""
    try:
        tifffile = importlib.import_module("tifffile")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tifffile is required to read TIFF stacks. Install it with 'pip install tifffile'."
        ) from exc

    with tifffile.TiffFile(imname) as tiff:
        data = tiff.asarray()
        axes = tiff.series[0].axes if tiff.series else ""

    data = np.asarray(data)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if data.ndim != 3:
        raise ValueError(
            f"Expected a grayscale TIFF stack with 3 dimensions, got {data.shape} "
            f"(axes={axes!r})."
        )
    return data


def save_imagej_tiff(imname, data, *, axes=None, dtype=np.float32):
    """Save an array in a TIFF format that ImageJ opens as a stack."""
    try:
        tifffile = importlib.import_module("tifffile")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tifffile is required to write TIFF stacks. Install it with 'pip install tifffile'."
        ) from exc

    output = np.asarray(data)
    if output.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D image or 3D stack, got {output.shape}.")
    if axes is None:
        axes = "YX" if output.ndim == 2 else "ZYX"
    if len(axes) != output.ndim:
        raise ValueError(f"axes={axes!r} does not match data shape {output.shape}.")

    tifffile.imwrite(
        Path(imname),
        output.astype(dtype, copy=False),
        imagej=True,
        metadata={"axes": axes},
    )

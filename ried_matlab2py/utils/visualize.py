"""Visualization helpers for raw and RIED-reconstructed image data."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import convolve2d

from .generate_rsf import generate_rsf
from .percennorm import percennorm


def _filter_image(image, rsf):
	"""Apply MATLAB ``imfilter``-like symmetric filtering to a 2D image."""
	return convolve2d(image, rsf, mode="same", boundary="symm")


def _key_colormap(name, positions, red, green, blue):
	if name.lower() == "green":
		cmap_name = "ried-green"
	elif name.lower() == "yellowhot":
		cmap_name = "ried-yellowhot"
	else:
		raise ValueError("Unsupported colormap. Use 'Green' or 'Yellowhot'.")
	return LinearSegmentedColormap.from_list(
		cmap_name,
		list(zip(positions, zip(red, green, blue))),
		N=256,
	)


def green_colormap():
	return _key_colormap(
		"green",
		[0, 0.12, 0.33, 0.60, 0.83, 1.00],
		[0, 0.03, 0.07, 0.14, 0.20, 0.85],
		[0, 0.18, 0.37, 0.64, 0.89, 1.00],
		[0, 0.02, 0.02, 0.05, 0.08, 0.08],
	)


def yellowhot_colormap():
	return _key_colormap(
		"yellowhot",
		[0, 0.31, 0.50, 0.70, 0.87, 1.00],
		[0, 0.73, 0.90, 1.00, 1.00, 1.00],
		[0, 0.55, 0.80, 1.00, 1.00, 1.00],
		[0, 0.00, 0.00, 0.09, 0.60, 1.00],
	)


def visualize(imgstack, RIEDrecon, baseline=0, rsf=1, pcn=(1, 100), cmap_fun="Green", show=True):
	"""Display raw summed data beside a RIED reconstruction.

	Parameters follow the MATLAB ``visualize`` function. ``baseline`` may be
	a scalar or an image matching the raw projection. Set ``show=False`` for
	headless use; the function then returns the figure and processed images.
	"""
	stack = np.asarray(imgstack, dtype=np.float64)
	if stack.ndim != 3:
		raise ValueError("imgstack must be a 3D array of image frames.")
	reconstruction = np.asarray(RIEDrecon, dtype=np.float64)
	if reconstruction.ndim != 2:
		raise ValueError("RIEDrecon must be a 2D array.")
	if len(pcn) != 2:
		raise ValueError("pcn must contain lower and upper percentile limits.")

	rsf_kernel = generate_rsf(rsf, None)
	raw = _filter_image(np.mean(stack, axis=2), rsf_kernel)
	raw = np.maximum(raw - baseline, 0)
	raw = percennorm(raw, 1, 100)
	reconstruction = percennorm(
		_filter_image(reconstruction, rsf_kernel), pcn[0], pcn[1]
	)

	if cmap_fun.lower() == "green":
		cmap = green_colormap()
	elif cmap_fun.lower() == "yellowhot":
		cmap = yellowhot_colormap()
	else:
		raise ValueError("Unsupported colormap. Use 'Green' or 'Yellowhot'.")

	import matplotlib.pyplot as plt

	figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
	axes[0].imshow(raw, vmin=0, vmax=1, cmap=cmap)
	axes[0].set_title("Raw summed data")
	axes[1].imshow(reconstruction, vmin=0, vmax=1, cmap=cmap)
	axes[1].set_title("RIED")
	for axis in axes:
		axis.set_axis_off()
	if show:
		plt.show()
	return figure, axes, raw, reconstruction

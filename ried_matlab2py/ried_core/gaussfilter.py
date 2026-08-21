from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

def fspecial_gaussian(N, sigma):
    """On the documents of MATLAB, they said this: 
    h = fspecial("gaussian",hsize,sigma) returns a rotationally symmetric Gaussian lowpass filter of size hsize with standard deviation sigma. 
    Not recommended. Use imgaussfilt or imgaussfilt3 instead.
    So I will check on it later! - T1E"""

    siz = (N - 1) / 2.0
    ax = np.arange(N) - siz
    x, y = np.meshgrid(ax, ax)
    arg = -(x ** 2 + y ** 2) / (2.0 * sigma ** 2)
    h = np.exp(arg)
    h[h < np.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h

def imfilter(img, kernel):

    N = kernel.shape[0]
    c0 = (N - 1) // 2  # 0-based kernel center, matches MATLAB imfilter centering
    pad_before = c0
    pad_after = N - 1 - c0
    padded = np.pad(img, ((pad_before, pad_after), (pad_before, pad_after)), mode='edge')
    # The Gaussian kernel is symmetric under 180-degree rotation, so
    # convolution and correlation coincide here.
    out = fftconvolve(padded, kernel, mode='valid')
    return out


def gaussfilter(data, sigma):

    data = np.asarray(data, dtype=float)
    N = int(np.ceil(sigma * 7))
    gauf = fspecial_gaussian(N, sigma)

    if data.ndim == 2:
        return imfilter(data, gauf)

    if data.ndim != 3:
        raise ValueError('gaussfilter only supports 2D or 3D input data.')

    data_gauss = np.zeros_like(data)
    for i in range(data.shape[2]):
        data_gauss[:, :, i] = imfilter(data[:, :, i], gauf)
    return data_gauss


if __name__ == '__main__':
    rng = np.random.default_rng(0)

    img2d = rng.random((40, 35))
    out2d = gaussfilter(img2d, 1.5)
    print('2D:', img2d.shape, '->', out2d.shape)
    assert out2d.shape == img2d.shape

    stack3d = rng.random((30, 28, 5))
    out3d = gaussfilter(stack3d, 2.0)
    print('3D:', stack3d.shape, '->', out3d.shape)
    assert out3d.shape == stack3d.shape

    # A smoothed constant image should remain (almost) exactly constant
    const_img = np.full((25, 25), 3.7)
    out_const = gaussfilter(const_img, 1.2)
    assert np.allclose(out_const, 3.7, atol=1e-8)

    print('All self-tests passed.')
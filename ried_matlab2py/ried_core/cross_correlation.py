from __future__ import annotations

import numpy as np


def cross_correlation(data, order, offset):

    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError('cross_correlation expects a 3D (nx, ny, nt) input array.')

    nx, ny, nt = data.shape
    output = np.zeros((nx * 2, ny * 2))

    cdata = np.abs(data - offset * np.mean(data, axis=2, keepdims=True)) ** order

    # "odd row, even col" (1-based) -- vertical-neighbor correlation
    output[1:2 * nx - 1:2, 0:2 * ny:2] = np.mean(
        cdata[0:nx - 1, :, :] * cdata[1:nx, :, :], axis=2
    )

    # "even row, odd col" (1-based) -- horizontal-neighbor correlation
    output[0:2 * nx:2, 1:2 * ny - 1:2] = np.mean(
        cdata[:, 0:ny - 1, :] * cdata[:, 1:ny, :], axis=2
    )

    # "odd row, odd col" interior (1-based) -- averaged diagonal-neighbor correlation
    output[1:2 * nx - 1:2, 1:2 * ny - 1:2] = (
        np.mean(cdata[0:nx - 1, 0:ny - 1, :] * cdata[1:nx, 1:ny, :], axis=2)
        + np.mean(cdata[0:nx - 1, 1:ny, :] * cdata[1:nx, 0:ny - 1, :], axis=2)
    ) / 2

    # "even row, even col" (1-based) -- original pixel grid, lag-1 temporal autocorrelation
    output[0:2 * nx:2, 0:2 * ny:2] = np.mean(
        cdata[:, :, 0:nt - 1] * cdata[:, :, 1:nt], axis=2
    )

    return output


if __name__ == '__main__':
    rng = np.random.default_rng(0)

    data = rng.random((12, 9, 20))
    out = cross_correlation(data, order=2.0, offset=1.0)
    print('shape:', data.shape, '->', out.shape)
    assert out.shape == (24, 18)

    # Non-square, small stack
    data2 = rng.random((5, 8, 4))
    out2 = cross_correlation(data2, order=1.5, offset=0.8)
    assert out2.shape == (10, 16)

    # Single-frame edge case: temporal autocorr positions should be NaN,
    # everything else finite.
    data3 = rng.random((4, 4, 1))
    with np.errstate(invalid='ignore'):
        out3 = cross_correlation(data3, order=2.0, offset=1.0)
    assert np.all(np.isnan(out3[0::2, 0::2]))
    assert np.all(np.isfinite(out3[1:-1:2, 0::2]))
    assert np.all(np.isfinite(out3[0::2, 1:-1:2]))

    print('All self-tests passed.')
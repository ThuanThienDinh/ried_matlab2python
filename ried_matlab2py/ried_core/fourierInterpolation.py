"""
fourierInterpolation.py

Python translation of fourierInterpolation.m (RIEDm / fSOFI toolkit).

Interpolation of a 2D or 3D input image using zero padding in the Fourier
domain. The input data can optionally be mirrored along the lateral/axial
(or both) dimensions to make the borders periodic, which reduces artifacts
at the image edges (see fSOFI publication).

Original MATLAB author: Simon Christoph Stein (scstein@phys.uni-goettingen.de), 2017
Python translation: faithful line-by-line port of the original algorithm.

Notes on the translation
-------------------------
- MATLAB is 1-based with inclusive ranges; all indexing below has been
  converted to 0-based, half-open Python/NumPy slicing.
- MATLAB's `round` rounds half-away-from-zero; NumPy's `np.round` uses
  round-half-to-even. Since inputs here are essentially always integers,
  this practically never matters, but `_matlab_round` is provided for
  strict fidelity.
- MATLAB's `padarray(..., 'symmetric', 'pre')` followed by a second
  `padarray(..., 'symmetric', 'post')` call is order-dependent (the second
  call mirrors the *already pre-padded* array). This is reproduced exactly
  below via two sequential `np.pad(..., mode='symmetric')` calls.
- The original file also defines a MATLAB-native `interpft3D` helper, but
  it is dead code (only referenced in comments, never called) and is
  therefore omitted here.
- In `fInterp_2D`/`fInterp_3D`, the original MATLAB `incr` handling has a
  latent bug: in the "downsampling" (`newsz < sz`) branch it accidentally
  overwrites the whole `incr` vector instead of a single element, which
  would raise an index-out-of-bounds error in MATLAB if more than one
  dimension is downsampled. Since `itp_fac` is documented/intended to be
  an interpolation (i.e. upsampling, >= 1) factor, this branch is not
  exercised in normal use. This translation implements the clearly
  intended per-dimension behavior instead of reproducing the crash.
"""

from __future__ import annotations

import numpy as np


def matlab_round(x):
    
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def fourier_interpolation(img, itp_fac, mirror_mode=None):
    """
    Interpolation of a 2D or 3D input image using zero padding in the
    Fourier domain.

    Parameters
    ----------
    img : ndarray (2D or 3D)
        Input image / volume.
    itp_fac : scalar or 1D array-like
        Interpolation factor along each dimension, e.g. [ipX, ipY] or
        [ipX, ipY, ipZ]. If a single number is given, the same factor is
        used for all dimensions. The output is of size
        itp_fac * img.shape (rounded).
    mirror_mode : {'none', 'lateral', 'axial', 'both'}, optional
        Whether to use periodic mirroring of the input data before
        interpolation (see fSOFI publication). Padding prevents artifacts
        from non-periodic borders and is essential if a low number of
        pixels is available along a specific dimension. Defaults to
        'none'.

    Returns
    -------
    ndarray
        Interpolated image / volume.
    """
    img = np.asarray(img)
    ndim = img.ndim
    if ndim not in (2, 3):
        raise ValueError('fourier_interpolation only supports 2D or 3D input images.')

    itp_fac = np.atleast_1d(np.asarray(itp_fac, dtype=float)).ravel()

    if itp_fac.size not in (1, ndim):
        raise ValueError(
            f'{itp_fac.size} interpolation factors specified. Give either one '
            f'for all dimensions or one per dimension!'
        )

    # If all interpolation factors are 1, skip the interpolation
    if np.all(itp_fac == 1):
        return img

    if itp_fac.size == 1:
        itp_fac = np.repeat(itp_fac, ndim)

    # for interpolation factors of 1, we perform neither padding nor interpolation
    noip = (itp_fac == 1)

    if mirror_mode is None:
        mirror_mode = 'none'

    input_sz = np.array(img.shape)

    # Starting index to cut out from upsampled periodic image
    sz = input_sz.copy()
    sz = sz - (1 - (sz % 2))  # always reduces sz to the largest odd value <= input_sz
    idx = (sz + 1) // 2 + 1 + (itp_fac - 1) * (sz // 2)  # 1-based, as in MATLAB
    idx0 = (idx - 1).astype(int)  # 0-based

    def get_valid_part(out_img):
        """Cut out the relevant (non-mirrored) part of a padded+interpolated image."""
        doip = ~noip
        if ndim == 2:

            if noip[0] and noip[1]:
                return out_img
            elif noip[0] and doip[1]:
                return out_img[:, idx0[1]:idx0[1] + int(itp_fac[1] * input_sz[1])]
            elif doip[0] and noip[1]:
                return out_img[idx0[0]:idx0[0] + int(itp_fac[0] * input_sz[0]), :]
            else:  # doip[0] and doip[1]
                return out_img[idx0[0]:idx0[0] + int(itp_fac[0] * input_sz[0]),
                                idx0[1]:idx0[1] + int(itp_fac[1] * input_sz[1])]

        if ndim == 3:

            if mirror_mode == 'lateral':
                if noip[0] and noip[1]:
                    return out_img
                elif noip[0] and doip[1]:
                    return out_img[:, idx0[1]:idx0[1] + int(itp_fac[1] * input_sz[1]), :]
                elif doip[0] and noip[1]:
                    return out_img[idx0[0]:idx0[0] + int(itp_fac[0] * input_sz[0]), :, :]
                else:  # doip[0] and doip[1]
                    return out_img[idx0[0]:idx0[0] + int(itp_fac[0] * input_sz[0]),
                                    idx0[1]:idx0[1] + int(itp_fac[1] * input_sz[1]), :]

            elif mirror_mode == 'axial':
                if doip[2]:
                    return out_img[:, :, idx0[2]:idx0[2] + int(itp_fac[2] * input_sz[2])]
                else:
                    return out_img

            elif mirror_mode == 'both':
                if noip[2]:  # No z-interpolation (with padding)
                    if noip[0] and noip[1]:
                        return out_img
                    elif noip[0] and doip[1]:
                        return out_img[:, idx0[1]:idx0[1] + int(itp_fac[1] * input_sz[1]), :]
                    elif doip[0] and noip[1]:
                        return out_img[idx0[0]:idx0[0] + int(itp_fac[0] * input_sz[0]), :, :]
                    else:  # doip[0] and doip[1]
                        return out_img[idx0[0]:idx0[0] + int(itp_fac[0] * input_sz[0]),
                                        idx0[1]:idx0[1] + int(itp_fac[1] * input_sz[1]), :]
                else:  # With z-interpolation
                    if noip[0] and noip[1]:
                        return out_img[:, :, idx0[2]:idx0[2] + int(itp_fac[2] * input_sz[2])]
                    elif noip[0] and doip[1]:
                        return out_img[:, idx0[1]:idx0[1] + int(itp_fac[1] * input_sz[1]),
                                        idx0[2]:idx0[2] + int(itp_fac[2] * input_sz[2])]
                    elif doip[0] and noip[1]:
                        return out_img[idx0[0]:idx0[0] + int(itp_fac[0] * input_sz[0]), :,
                                        idx0[2]:idx0[2] + int(itp_fac[2] * input_sz[2])]
                    else:  # doip[0] and doip[1]
                        return out_img[idx0[0]:idx0[0] + int(itp_fac[0] * input_sz[0]),
                                        idx0[1]:idx0[1] + int(itp_fac[1] * input_sz[1]),
                                        idx0[2]:idx0[2] + int(itp_fac[2] * input_sz[2])]
            else:
                raise ValueError(f"Unknown padding option '{mirror_mode}'.")

    def pad_symmetric_pre_post(arr, pre, post):
        """Replicates two sequential MATLAB padarray('symmetric') calls (pre then post)."""
        pre = [int(p) for p in pre]
        post = [int(p) for p in post]
        arr = np.pad(arr, [(p, 0) for p in pre], mode='symmetric')
        arr = np.pad(arr, [(0, p) for p in post], mode='symmetric')
        return arr

    # ------------------------------------------------------------------ #
    if ndim == 2:
        if mirror_mode == 'none':
            newsz = matlab_round(itp_fac * np.array(img.shape)).astype(int)
            return _finterp_2d(img, newsz)

        elif mirror_mode == 'lateral':
            padsize = np.array([img.shape[0] / 2.0, img.shape[1] / 2.0])
            padsize[noip] = 0
            pre = np.ceil(padsize)
            post = np.floor(padsize)
            img = pad_symmetric_pre_post(img, pre, post)

            newsz = matlab_round(itp_fac * np.array(img.shape) - (itp_fac - 1)).astype(int)
            img = _finterp_2d(img, newsz)
            return get_valid_part(img)

        elif mirror_mode == 'axial':
            raise ValueError("Padding 'axial' only possible for 3D data.")
        elif mirror_mode == 'both':
            raise ValueError("Padding 'both' only possible for 3D data.")
        else:
            raise ValueError(f"Unknown padding option '{mirror_mode}'.")

    else:  # ndim == 3
        if mirror_mode == 'none':
            newsz = matlab_round(itp_fac * np.array(img.shape)).astype(int)
            return _finterp_3d(img, newsz)

        elif mirror_mode == 'lateral':
            padsize = np.array([img.shape[0] / 2.0, img.shape[1] / 2.0, 0.0])
            padsize[noip] = 0
            pre = np.ceil(padsize)
            post = np.floor(padsize)
            img = pad_symmetric_pre_post(img, pre, post)

            newsz = matlab_round(np.array([
                itp_fac[0] * img.shape[0] - (itp_fac[0] - 1),
                itp_fac[1] * img.shape[1] - (itp_fac[1] - 1),
                itp_fac[2] * img.shape[2],
            ])).astype(int)
            img = _finterp_3d(img, newsz)
            return get_valid_part(img)

        elif mirror_mode == 'axial':
            padsize = np.array([0.0, 0.0, img.shape[2] / 2.0])
            padsize[noip] = 0
            pre = np.ceil(padsize)
            post = np.floor(padsize)
            img = pad_symmetric_pre_post(img, pre, post)

            newsz = matlab_round(np.array([
                itp_fac[0] * img.shape[0],
                itp_fac[1] * img.shape[1],
                itp_fac[2] * img.shape[2] - (itp_fac[2] - 1),
            ])).astype(int)
            img = _finterp_3d(img, newsz)
            return get_valid_part(img)

        elif mirror_mode == 'both':
            padsize = np.array(img.shape) / 2.0
            padsize[noip] = 0
            pre = np.ceil(padsize)
            post = np.floor(padsize)
            img = pad_symmetric_pre_post(img, pre, post)

            newsz = matlab_round(itp_fac * np.array(img.shape) - (itp_fac - 1)).astype(int)
            img = _finterp_3d(img, newsz)
            return get_valid_part(img)

        else:
            raise ValueError(f"Unknown padding option '{mirror_mode}'.")


# ---------------------------------------------------------------------- #
# Core FFT-based zero-padding interpolation (equivalent of MATLAB interpft
# applied along each dimension, but done in one shot for speed).
# ---------------------------------------------------------------------- #

def _finterp_2d(img, newsz):
    """Fourier interpolation of a 2D image to new size newsz = [nx, ny]."""
    img = np.asarray(img)
    sz = np.array(img.shape)
    newsz = np.array(newsz, dtype=int).copy()

    if np.any(newsz == 0):
        return np.zeros((0, 0))

    isgreater = newsz >= sz
    incr = np.ones(2, dtype=int)
    for i_dim in range(2):
        if isgreater[i_dim]:
            incr[i_dim] = 1
        else:
            incr[i_dim] = sz[i_dim] // newsz[i_dim] + 1
            newsz[i_dim] = incr[i_dim] * newsz[i_dim]

    img_ip = np.zeros(tuple(newsz), dtype=complex)
    nyqst = (sz + 1 + 1) // 2  # == ceil((sz+1)/2)

    # multiplicative factor conserves the counts at the original positions
    X = (newsz[0] / sz[0]) * (newsz[1] / sz[1]) * np.fft.fft2(img)

    hic = sz - nyqst  # number of "high"-frequency bins per dimension

    # zero padding: copy all 4 corners of the spectrum
    img_ip[0:nyqst[0], 0:nyqst[1]] = X[0:nyqst[0], 0:nyqst[1]]  # xl, yl
    img_ip[newsz[0] - hic[0]:newsz[0], 0:nyqst[1]] = X[nyqst[0]:sz[0], 0:nyqst[1]]  # xh, yl
    img_ip[0:nyqst[0], newsz[1] - hic[1]:newsz[1]] = X[0:nyqst[0], nyqst[1]:sz[1]]  # xl, yh
    img_ip[newsz[0] - hic[0]:newsz[0], newsz[1] - hic[1]:newsz[1]] = \
        X[nyqst[0]:sz[0], nyqst[1]:sz[1]]  # xh, yh

    rm = sz % 2
    if rm[0] == 0 and newsz[0] != sz[0]:
        img_ip[nyqst[0] - 1, :] = img_ip[nyqst[0] - 1, :] / 2
        img_ip[nyqst[0] - 1 + (newsz[0] - sz[0]), :] = img_ip[nyqst[0] - 1, :]
    if rm[1] == 0 and newsz[1] != sz[1]:
        img_ip[:, nyqst[1] - 1] = img_ip[:, nyqst[1] - 1] / 2
        img_ip[:, nyqst[1] - 1 + (newsz[1] - sz[1])] = img_ip[:, nyqst[1] - 1]

    img_ip = np.real(np.fft.ifft2(img_ip))
    # Skip points if necessary
    img_ip = img_ip[0:newsz[0]:incr[0], 0:newsz[1]:incr[1]]
    return img_ip


def _finterp_3d(img, newsz):
    """Fourier interpolation of a 3D image to new size newsz = [nx, ny, nz]."""
    img = np.asarray(img)
    sz = np.array(img.shape)
    newsz = np.array(newsz, dtype=int).copy()

    if np.any(newsz == 0):
        return np.zeros((0, 0, 0))

    isgreater = newsz >= sz
    incr = np.ones(3, dtype=int)
    for i_dim in range(3):
        if isgreater[i_dim]:
            incr[i_dim] = 1
        else:
            incr[i_dim] = sz[i_dim] // newsz[i_dim] + 1
            newsz[i_dim] = incr[i_dim] * newsz[i_dim]

    img_ip = np.zeros(tuple(newsz), dtype=complex)
    nyqst = (sz + 1 + 1) // 2  # == ceil((sz+1)/2)

    # multiplicative factor conserves the counts at the original positions
    X = (newsz[0] / sz[0]) * (newsz[1] / sz[1]) * (newsz[2] / sz[2]) * np.fft.fftn(img)

    hic = sz - nyqst  # number of "high"-frequency bins per dimension

    def lo(d):
        return slice(0, nyqst[d])

    def hi_dst(d):
        return slice(newsz[d] - hic[d], newsz[d])

    def lo_src(d):
        return slice(0, nyqst[d])

    def hi_src(d):
        return slice(nyqst[d], sz[d])

    # zero padding, copy all 8 octants of the spectrum
    img_ip[lo(0), lo(1), lo(2)] = X[lo_src(0), lo_src(1), lo_src(2)]              # xl, yl, zl
    img_ip[hi_dst(0), lo(1), lo(2)] = X[hi_src(0), lo_src(1), lo_src(2)]          # xh, yl, zl
    img_ip[lo(0), hi_dst(1), lo(2)] = X[lo_src(0), hi_src(1), lo_src(2)]          # xl, yh, zl
    img_ip[lo(0), lo(1), hi_dst(2)] = X[lo_src(0), lo_src(1), hi_src(2)]          # xl, yl, zh
    img_ip[hi_dst(0), hi_dst(1), lo(2)] = X[hi_src(0), hi_src(1), lo_src(2)]      # xh, yh, zl
    img_ip[hi_dst(0), lo(1), hi_dst(2)] = X[hi_src(0), lo_src(1), hi_src(2)]      # xh, yl, zh
    img_ip[lo(0), hi_dst(1), hi_dst(2)] = X[lo_src(0), hi_src(1), hi_src(2)]      # xl, yh, zh
    img_ip[hi_dst(0), hi_dst(1), hi_dst(2)] = X[hi_src(0), hi_src(1), hi_src(2)]  # xh, yh, zh

    rm = sz % 2
    if rm[0] == 0 and newsz[0] != sz[0]:
        img_ip[nyqst[0] - 1, :, :] = img_ip[nyqst[0] - 1, :, :] / 2
        img_ip[nyqst[0] - 1 + (newsz[0] - sz[0]), :, :] = img_ip[nyqst[0] - 1, :, :]
    if rm[1] == 0 and newsz[1] != sz[1]:
        img_ip[:, nyqst[1] - 1, :] = img_ip[:, nyqst[1] - 1, :] / 2
        img_ip[:, nyqst[1] - 1 + (newsz[1] - sz[1]), :] = img_ip[:, nyqst[1] - 1, :]
    if rm[2] == 0 and newsz[2] != sz[2]:
        img_ip[:, :, nyqst[2] - 1] = img_ip[:, :, nyqst[2] - 1] / 2
        img_ip[:, :, nyqst[2] - 1 + (newsz[2] - sz[2])] = img_ip[:, :, nyqst[2] - 1]

    img_ip = np.real(np.fft.ifftn(img_ip))
    # Skip points if necessary
    img_ip = img_ip[0:newsz[0]:incr[0], 0:newsz[1]:incr[1], 0:newsz[2]:incr[2]]
    return img_ip
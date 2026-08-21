"""
background_noise_estimation.py

Python translation of background_noise_estimation.m.

    function Background = background_noise_estimation(imgs,th,dlevel,wavename,iter)
    ...
    [x,y,~]=size(imgs);
    if x<y
        imgs=padarray(imgs,[max(x,y)-size(imgs,1),max(x,y)-size(imgs,2),0],'post','symmetric');
    end
    Background = zeros(size(imgs),'single');
    for frames = 1: size(imgs,3)
        initial = imgs(:,:,frames);
        res = initial;
        for ii = 1:iter
            [m,n] = wavedec2(res,dlevel,wavename);
            vec = zeros(size(m));
            vec(1:n(1)*n(1)*1) = m(1:n(1)*n(1)*1);
            Biter =  waverec2(vec,n,wavename);
            if th > 0
                eps = sqrt(abs(res))/2;
                ind = initial>(Biter+eps);
                res(ind) = Biter(ind)+eps(ind);
                [m,n] = wavedec2(res,dlevel,wavename);
                vec = zeros(size(m));
                vec(1:n(1)*n(1)*1) = m(1:n(1)*n(1)*1);
                Biter =  waverec2(vec,n,wavename);
            end
        end
        Background(:,:,frames) = Biter;
    end
    Background=Background(1:x,1:y,:);

Estimates a smooth background for each frame of an image stack by
repeatedly (1) taking only the coarsest-level wavelet *approximation*
coefficients (i.e. a heavily low-pass-filtered version of the image,
with all detail/high-frequency coefficients zeroed) as the background
estimate, and (2) clamping bright foreground pixels down towards that
background estimate (plus a Poisson-like noise margin `eps`) before
re-estimating, so that real signal doesn't bias the background upward.

Dependencies
------------
Requires PyWavelets (`pip install PyWavelets`).

Notes on the translation
-------------------------
- `[m,n] = wavedec2(res,dlevel,wavename)` returns a flat coefficient
  vector `m` and a bookkeeping matrix `n` (traditionally called `S`).
  `n(1)` (MATLAB linear/column-major indexing into the bookkeeping
  matrix) is `S(1,1)`, the number of rows of the coarsest-level
  approximation block. Because the image is padded to be square before
  this loop runs, the approximation block is always square, so
  `n(1)*n(1)` is exactly the count of approximation coefficients, which
  are always stored first in `m`. `vec(1:n(1)*n(1)) = m(1:n(1)*n(1))`
  therefore keeps only the approximation coefficients and zeroes every
  detail coefficient at every level -- equivalent to reconstructing from
  `pywt.wavedec2` output with every detail tuple replaced by zeros while
  keeping the approximation array (`coeffs[0]`) unchanged. This is
  implemented in `_wavelet_lowpass` below.
- MATLAB's default wavelet extension/boundary mode (via `dwtmode`) is
  `'sym'` (half-point symmetric), which is the same convention as
  PyWavelets' default `mode='symmetric'`. `mode='symmetric'` is passed
  explicitly to `wavedec2`/`waverec2` below to make this match explicit
  rather than relying on defaults.
- `waverec2` in PyWavelets can occasionally return an array 1 pixel
  larger than the input along a dimension, depending on filter length
  and decomposition level parity; the result is cropped back to the
  original (padded, square) frame size to guard against this.
- The local variable named `eps` in the original MATLAB code shadows
  MATLAB's builtin `eps` function; here it is named `eps_margin` to
  avoid any confusion. Likewise the `iter` parameter (which shadows
  MATLAB's `iter` function / Python's builtin `iter`) is renamed
  `n_iter`, keeping the same positional order and defaults.
- `padarray(imgs, [...], 'post', 'symmetric')` pads only the trailing
  edge of dimensions 1 and 2 (never dimension 3) using edge-mirrored
  ("symmetric") padding; this is reproduced with
  `np.pad(..., mode='symmetric')`.
- I was unable to install PyWavelets or MATLAB in the environment used
  to prepare this translation, so this could not be executed/verified
  numerically end-to-end the way the other translated functions were.
  Please test this one directly against your MATLAB reference output
  before relying on it.
"""

from __future__ import annotations

import numpy as np

try:
    import pywt
except ImportError as _e:  # pragma: no cover
    pywt = None
    _PYWT_IMPORT_ERROR = _e


def wavelet_lowpass(img, dlevel, wavename):
    """
    Python has waverec2 function similar to wavedec2 in MATLAB - T1E
    """
    if pywt is None:  # pragma: no cover
        raise ImportError(
            'background_noise_estimation requires PyWavelets. '
            'Install it with: pip install PyWavelets'
        ) from _PYWT_IMPORT_ERROR

    coeffs = pywt.wavedec2(img, wavename, mode='symmetric', level=dlevel)
    cA = coeffs[0]
    zeroed_details = [
        tuple(np.zeros_like(d) for d in detail_level)
        for detail_level in coeffs[1:]
    ]
    rec = pywt.waverec2([cA] + zeroed_details, wavename, mode='symmetric')
    # Guard against occasional +/-1 pixel size mismatch from waverec2.
    rec = rec[:img.shape[0], :img.shape[1]]
    return rec


def background_noise_estimation(imgs, th=1, dlevel=7, wavename='db6', n_iter=3):

    imgs = np.asarray(imgs, dtype=np.float64)
    was_2d = (imgs.ndim == 2)
    if was_2d:
        imgs = imgs[:, :, np.newaxis]
    if imgs.ndim != 3:
        raise ValueError('background_noise_estimation expects a 2D or 3D input array.')

    x, y, n_frames = imgs.shape  # original size, used for final cropping

    if x < y:
        pad_x = max(x, y) - x
        pad_y = max(x, y) - y
        imgs = np.pad(imgs, ((0, pad_x), (0, pad_y), (0, 0)), mode='symmetric')

    px, py, _ = imgs.shape
    background = np.zeros((px, py, n_frames), dtype=np.float32)

    for frame in range(n_frames):
        initial = imgs[:, :, frame]
        res = initial.copy()
        biter = None

        for _ii in range(n_iter):
            biter = wavelet_lowpass(res, dlevel, wavename)

            if th > 0:
                eps_margin = np.sqrt(np.abs(res)) / 2.0
                ind = initial > (biter + eps_margin)
                res = res.copy()
                res[ind] = biter[ind] + eps_margin[ind]
                biter = wavelet_lowpass(res, dlevel, wavename)

        background[:, :, frame] = biter

    background = background[:x, :y, :]
    if was_2d:
        background = background[:, :, 0]
    return background


if __name__ == '__main__':
    if pywt is None:
        print('PyWavelets is not installed in this environment; '
              'install it with `pip install PyWavelets` to run this self-test.')
    else:
        rng = np.random.default_rng(0)

        # Non-square stack, to exercise the padding-to-square branch (x < y).
        stack = rng.random((20, 32, 3)).astype(np.float32) * 100

        bg = background_noise_estimation(stack, th=1, dlevel=3, wavename='db6', n_iter=2)
        print('3D:', stack.shape, '->', bg.shape)
        assert bg.shape == stack.shape
        assert bg.dtype == np.float32

        # A smooth background should have (much) lower variance than the
        # noisy input, and should not itself contain the sharp spikes we
        # add below.
        smooth = np.ones((20, 20)) * 50.0
        smooth_noisy = smooth.copy()
        smooth_noisy[5, 5] = 500.0  # bright spike (simulated foreground signal)
        bg2 = background_noise_estimation(smooth_noisy, th=1, dlevel=3, wavename='db6', n_iter=3)
        print('spike suppressed:', bg2[5, 5] < 500.0, 'value at spike:', bg2[5, 5])

        # 2D input in, 2D output out.
        img2d = rng.random((16, 16)).astype(np.float32)
        bg2d = background_noise_estimation(img2d, dlevel=2, n_iter=1)
        assert bg2d.shape == img2d.shape

        print('Self-tests ran without errors (shapes/dtype OK). '
              'This has NOT been cross-checked against MATLAB output -- '
              'please validate against your reference before relying on it.')
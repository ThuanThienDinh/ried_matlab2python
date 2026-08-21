from __future__ import annotations
import numpy as np
import sys
from pathlib import Path
try:
    from .RLdeconv import RLdeconv
except ImportError:
    from RLdeconv import RLdeconv

try:
    from utils.forward_diff import forward_diff
    from utils.backward_diff import backward_diff
except ImportError:
    try:
        from utils.forward_diff import forward_diff
        from utils.backward_diff import backward_diff
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.forward_diff import forward_diff
        from utils.backward_diff import backward_diff

def _build_frefft(sizex, zbei):

    def fft_pow2(kernel):
        F = np.fft.fftn(kernel, s=sizex)
        return F * np.conj(F)

    # Pure 2nd differences along each axis
    k_d1 = np.array([1, -2, 1], dtype=np.float64).reshape(3, 1, 1)
    k_d2 = np.array([1, -2, 1], dtype=np.float64).reshape(1, 3, 1)
    k_d3 = np.array([1, -2, 1], dtype=np.float64).reshape(1, 1, 3)  # ztiduzz

    # Mixed 2nd differences (cross terms)
    k_d12 = np.array([[1, -1], [-1, 1]], dtype=np.float64).reshape(2, 2, 1)
    k_d13 = np.array([1, -1, -1, 1], dtype=np.float64).reshape(2, 1, 2)  # varies dim1 & dim3
    k_d23 = np.array([1, -1, -1, 1], dtype=np.float64).reshape(1, 2, 2)  # varies dim2 & dim3

    Frefft = fft_pow2(k_d2)                     # matches original 1st term ([1 -2 1] row vec)
    Frefft = Frefft + fft_pow2(k_d1)             # matches original 2nd term ([1;-2;1] col vec)
    Frefft = Frefft + (zbei ** 2) * fft_pow2(k_d3)      # ztiduzz term
    Frefft = Frefft + 2 * fft_pow2(k_d12)               # [1 -1;-1 1] term
    Frefft = Frefft + 2 * zbei * fft_pow2(k_d13)         # ztiduyz-equivalent term
    Frefft = Frefft + 2 * zbei * fft_pow2(k_d23)         # ztiduxz-equivalent term

    return Frefft


def Spasedeconv(input, mu, sigma_t, l10, iter_Bregman, psf, iter, scale):

    lamda = 1
    siranu = mu
    zbei = sigma_t

    input = np.asarray(input, dtype=np.float32)
    if input.ndim == 2:
        input = input[:, :, np.newaxis]
    sx, sy, _ = input.shape

    ind_sam = 1
    sy_full = sy * ind_sam
    sx_full = sx * ind_sam

    y = np.zeros((sx_full, sy_full, input.shape[2]), dtype=np.float32)
    y[0:sx_full:ind_sam, 0:sy_full:ind_sam, :] = input

    sx, sy, sz = y.shape
    sizex = (sx, sy, sz)
    y_flag = y.shape[2]
    l1 = l10

    Frefft = _build_frefft(sizex, zbei)
    divide = ((siranu / lamda) + Frefft).astype(np.complex64)
    del Frefft

    b1 = np.zeros(sizex, dtype=np.float32)
    b2 = np.zeros(sizex, dtype=np.float32)
    b3 = np.zeros(sizex, dtype=np.float32)
    b4 = np.zeros(sizex, dtype=np.float32)
    b5 = np.zeros(sizex, dtype=np.float32)
    b6 = np.zeros(sizex, dtype=np.float32)
    b7 = np.zeros(sizex, dtype=np.float32)
    b8 = np.zeros(sizex, dtype=np.float32)  # unused, kept for structural fidelity
    x = np.zeros(sizex, dtype=np.float32)

    frac = (siranu / lamda) * y
    frac[frac < 0] = 0

    for ii in range(1, iter_Bregman + 1):
        frac = np.fft.fftn(frac)
        if ii > 1:
            x = np.real(np.fft.ifftn(frac / divide))
        else:
            x = np.real(np.fft.ifftn(frac / (siranu / lamda)))
        frac = (siranu / lamda) * y
        # frac(frac<0)=0;  # intentionally NOT re-applied inside the loop (see docstring)

        u = backward_diff(forward_diff(x, 1, 0), 1, 0)
        signd = np.abs(u + b1) - 1 / lamda
        signd[signd < 0] = 0
        signd = signd * np.sign(u + b1)
        d = signd
        b1 = b1 + (u - d)
        frac = frac + backward_diff(forward_diff(d - b1, 1, 0), 1, 0)

        u = backward_diff(forward_diff(x, 1, 1), 1, 1)
        signd = np.abs(u + b2) - 1 / lamda
        signd[signd < 0] = 0
        signd = signd * np.sign(u + b2)
        d = signd
        b2 = b2 + (u - d)
        frac = frac + backward_diff(forward_diff(d - b2, 1, 1), 1, 1)

        u = backward_diff(forward_diff(x, 1, 2), 1, 2)
        signd = np.abs(u + b3) - 1 / lamda
        signd[signd < 0] = 0
        signd = signd * np.sign(u + b3)
        d = signd
        b3 = b3 + (u - d)
        frac = frac + (zbei ** 2) * backward_diff(forward_diff(d - b3, 1, 2), 1, 2)

        u = forward_diff(forward_diff(x, 1, 0), 1, 1)
        signd = np.abs(u + b4) - 1 / lamda
        signd[signd < 0] = 0
        signd = signd * np.sign(u + b4)
        d = signd
        b4 = b4 + (u - d)
        frac = frac + 2 * backward_diff(backward_diff(d - b4, 1, 1), 1, 0)

        u = forward_diff(forward_diff(x, 1, 0), 1, 2)
        signd = np.abs(u + b5) - 1 / lamda
        signd[signd < 0] = 0
        signd = signd * np.sign(u + b5)
        d = signd
        b5 = b5 + (u - d)
        frac = frac + 2 * zbei * backward_diff(backward_diff(d - b5, 1, 2), 1, 0)

        u = forward_diff(forward_diff(x, 1, 1), 1, 2)
        signd = np.abs(u + b6) - 1 / lamda
        signd[signd < 0] = 0
        signd = signd * np.sign(u + b6)
        d = signd
        b6 = b6 + (u - d)
        frac = frac + 2 * zbei * backward_diff(backward_diff(d - b6, 1, 2), 1, 1)

        u = x
        signd = np.abs(u + b7) - 1 / lamda
        signd[signd < 0] = 0
        signd = signd * np.sign(u + b7)
        d = signd
        b7 = b7 + (u - d)
        frac = frac + l1 * (d - b7)

        x[x < 0] = 0

    x[x < 0] = 0
    x = x[:, :, :y_flag]
    cr = x.astype(np.float32)
    cr_max = cr.max()
    if cr_max > 0:
        cr = cr / cr_max

    psf_3d = psf[:, :, np.newaxis] if psf.ndim == 2 else psf
    output = np.abs(RLdeconv(cr, psf_3d ** scale, iter, 0))
    output_max = output.max()
    if output_max > 0:
        output = output / output_max
    return output

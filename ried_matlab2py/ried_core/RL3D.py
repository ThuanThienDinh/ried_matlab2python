import numpy as np
from typing import Union, Optional

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert a Point Spread Function (PSF) into an Optical Transfer Function (OTF).

    The PSF is zero-padded to match the target shape, centered, and then 
    transformed to the frequency domain using FFT.
    
    Parameters
    ----------
    psf : np.ndarray
        The point spread function kernel.
    shape : tuple
        The target output shape for the OTF.
    
    Returns
    -------
    np.ndarray
        Complex-valued OTF with the specified shape.
    
    Raises
    ------
    ValueError
        If PSF dimensions don't match target shape dimensions.
    TypeError
        If psf is not array-like or shape is not array-like.
    """
    try:
        psf = np.asarray(psf, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError("psf must be convertible to numpy array") from e
    
    try:
        shape = tuple(int(s) for s in shape)
    except (ValueError, TypeError) as e:
        raise TypeError("shape must be convertible to tuple of integers") from e
    
    if not all(s > 0 for s in shape):
        raise ValueError(f"All shape dimensions must be positive, got {shape}")

    if psf.ndim != len(shape):
        raise ValueError(
            f"PSF has {psf.ndim} dimensions "
            f"but the target shape has {len(shape)} dimensions."
        )

    psf_padded = np.zeros(shape, dtype=np.float64)
    slices = tuple(slice(0, min(psf.shape[i], shape[i])) for i in range(psf.ndim))
    psf_padded[slices] = psf[slices]

    for axis, size in enumerate(psf.shape):
        psf_padded = np.roll(psf_padded, -(size // 2), axis=axis)

    return np.fft.fftn(psf_padded)


def deblur_coreRL(yk, data, otf, xk, vk, iter_idx, xp):
    """
    Core Richardson-Lucy iteration step with acceleration.
    
    Performs one iteration of the Richardson-Lucy algorithm with 
    Nesterov-like acceleration to improve convergence speed.
    
    Parameters
    ----------
    yk : array
        Current deconvolved estimate.
    data : array
        Observed (blurred) data.
    otf : array
        Optical Transfer Function (in frequency domain).
    xk : array
        Previous deconvolved estimate (for acceleration).
    vk : array
        Acceleration term from previous iteration.
    iter_idx : int
        Current iteration index (1-based).
    xp : module
        Array module (numpy or cupy) for computation.
    
    Returns
    -------
    tuple
        (yk_new, xk_new, vk_new) - Updated estimates and acceleration term.
    """
    xk_update = xk.copy()
    
    # Forward convolution: estimate = PSF * yk
    estimate = xp.real(xp.fft.ifftn(otf * xp.fft.fftn(yk)))
    estimate = xp.maximum(estimate, 1e-5)
    
    # Compute error ratio
    ratio = data / estimate
    
    # Backward convolution: correction = PSF_conj * ratio
    correction = xp.real(xp.fft.ifftn(xp.conj(otf) * xp.fft.fftn(ratio)))
    
    # Update estimate with correction
    xk = xp.maximum(yk * correction, 1e-5)
    
    vk_update = vk.copy()
    vk = xk - yk

    # Compute acceleration parameter
    if iter_idx == 1:
        alpha = 0.0
    else:
        numerator = xp.sum(vk_update * vk)
        denominator = xp.sum(vk_update * vk_update) + np.finfo(np.float64).eps
        alpha = xp.clip(numerator / denominator, 0.0, 1.0)

    # Apply acceleration
    yk = xp.real(xp.maximum(xk + alpha * (xk - xk_update), 1e-5))

    return yk, xk, vk


def RL3D(data: np.ndarray, kernel: np.ndarray, iteration: Optional[int] = None, 
         gpu: Union[int, bool] = 0, iterations: Optional[int] = None) -> np.ndarray:
    """
    3D Richardson-Lucy deconvolution with optional GPU acceleration.
    
    Performs iterative deconvolution on 3D volumetric data using the 
    Richardson-Lucy algorithm with Nesterov-like acceleration.
    
    Parameters
    ----------
    data : np.ndarray
        3D observed (blurred) data volume. Must be a 3D array with positive values.
    kernel : np.ndarray
        3D point spread function (PSF) or kernel. Must be 3D and sum to positive value.
    iteration : int, optional
        Number of iterations. Must be >= 1. Alias for 'iterations'.
    gpu : int or bool, default=0
        GPU device ID (0+ for GPU, 0 for CPU). If truthy, uses GPU computation via CuPy.
    iterations : int, optional
        Number of iterations. Alternative to 'iteration' parameter. 
        Only one of 'iteration' or 'iterations' should be specified.
    
    Returns
    -------
    np.ndarray
        Deconvolved 3D data with same shape as input.
    
    Raises
    ------
    TypeError
        If data or kernel cannot be converted to arrays, or if iteration is not specified.
    ValueError
        If dimensions don't match, values are invalid, or both iteration parameters specified.
    ImportError
        If GPU requested but CuPy not installed.
    
    Examples
    --------
    >>> data = np.random.rand(32, 32, 32)
    >>> kernel = np.ones((3, 3, 3))
    >>> result = RL3D(data, kernel, iteration=10, gpu=0)
    """
    # Handle iteration parameter ambiguity
    if iteration is None:
        iteration = iterations
    elif iterations is not None and iterations != iteration:
        raise ValueError("Specify only one of 'iteration' or 'iterations'.")

    if iteration is None:
        raise TypeError("RL3D() missing required argument: 'iteration' or 'iterations'.")

    iteration = int(iteration)
    if iteration < 1:
        raise ValueError(f"iteration must be >= 1, got {iteration}.")

    # Setup array module (numpy or cupy)
    if gpu:
        try:
            import cupy as cp  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "GPU mode requires CuPy to be installed. "
                "Install with: pip install cupy-cuda11x (replace 11x with your CUDA version)"
            ) from exc
        xp = cp
    else:
        xp = np

    # Input validation and conversion
    try:
        data = np.asarray(data, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError("data must be convertible to numpy array") from e

    try:
        kernel = np.asarray(kernel, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError("kernel must be convertible to numpy array") from e

    # Dimension validation
    if data.ndim != 3:
        raise ValueError(f"RL3D expects data to be 3D, got {data.ndim}D array with shape {data.shape}.")

    if kernel.ndim != 3:
        raise ValueError(f"RL3D expects kernel to be 3D, got {kernel.ndim}D array with shape {kernel.shape}.")

    # Value validation
    if not np.all(np.isfinite(data)):
        raise ValueError("data contains non-finite values (NaN or Inf).")

    if not np.all(np.isfinite(kernel)):
        raise ValueError("kernel contains non-finite values (NaN or Inf).")

    if np.any(data < 0):
        raise ValueError("data contains negative values (expected non-negative).")

    if np.any(kernel < 0):
        raise ValueError("kernel contains negative values (expected non-negative).")

    kernel_sum = np.sum(kernel)
    if kernel_sum <= 0:
        raise ValueError(f"kernel sum must be > 0, got {kernel_sum}.")

    # Normalize kernel
    kernel = kernel / kernel_sum

    # Compute OTF (always on CPU for stability)
    otf = psf2otf(kernel, data.shape)

    # Transfer to GPU if needed
    if gpu:
        data = xp.asarray(data)
        otf = xp.asarray(otf)

    # Initialize algorithm state
    yk = data.copy()
    xk = xp.zeros_like(data)
    vk = xp.zeros_like(data)

    # Iterative deconvolution
    for iter_idx in range(1, iteration + 1):
        yk, xk, vk = deblur_coreRL(yk, data, otf, xk, vk, iter_idx, xp)

    # Convert back to CPU if using GPU
    if gpu:
        yk = xp.asnumpy(yk)

    return np.real(yk)
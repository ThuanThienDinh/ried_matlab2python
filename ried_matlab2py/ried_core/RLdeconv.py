"""Wrapper module for 3D Richardson-Lucy deconvolution."""

from typing import Union
import numpy as np
from .RL3D import RL3D


def RLdeconv(data: np.ndarray, kernel: np.ndarray, iteration: int, 
             gpu: Union[int, bool] = 0) -> np.ndarray:
    return RL3D(data, kernel, iteration=iteration, gpu=gpu)
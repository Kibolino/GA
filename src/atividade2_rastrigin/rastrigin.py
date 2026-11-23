from __future__ import annotations

import math

import numpy as np

A = 10.0
LIM_INF, LIM_SUP = -5.12, 5.12
DOIS_PI = 2.0 * math.pi


def rastrigin(x, a: float = A) -> np.ndarray | float:
    x = np.asarray(x, dtype=float)
    return a * x.shape[-1] + np.sum(x * x - a * np.cos(DOIS_PI * x), axis=-1)


def rastrigin_individuo(x, a: float = A) -> float:
    return a * len(x) + sum(xi * xi - a * math.cos(DOIS_PI * xi) for xi in x)

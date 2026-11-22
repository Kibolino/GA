"""Função de Rastrigin.

    f(x) = A n + sum_{i=1}^{n} [x_i^2 - A cos(2 pi x_i)],   A = 10,  x_i em [-5,12; 5,12]

Mínimo global f(0) = 0. Cada coordenada contribui com
g(x_i) = x_i^2 + A (1 - cos(2 pi x_i)) >= 0, que tem mínimos locais perto de
cada inteiro k com valor ~k^2. Assim, um mínimo local com m coordenadas em
+/-1 vale ~m, o que permite interpretar o valor final de f como "quantas
coordenadas ficaram presas em uma bacia vizinha".
"""
from __future__ import annotations

import math

import numpy as np

A = 10.0
LOW, UP = -5.12, 5.12
TWO_PI = 2.0 * math.pi


def rastrigin(x, a: float = A) -> np.ndarray | float:
    """Versão vetorizada: aceita array (..., n) e devolve (...)."""
    x = np.asarray(x, dtype=float)
    return a * x.shape[-1] + np.sum(x * x - a * np.cos(TWO_PI * x), axis=-1)


def rastrigin_py(x, a: float = A) -> float:
    """Versão em Python puro para um único indivíduo (mais rápida que numpy
    para vetores pequenos, que é o caso da avaliação individual no DEAP)."""
    return a * len(x) + sum(xi * xi - a * math.cos(TWO_PI * xi) for xi in x)

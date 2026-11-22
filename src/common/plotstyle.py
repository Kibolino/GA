"""Estilo único dos gráficos (matplotlib) usado nas duas atividades.

As figuras são dimensionadas para o artigo em duas colunas (IEEE): uma coluna
tem ~3,5 in de largura e a página inteira ~7,1 in. Cada figura é salva em PDF
(vetorial, usada no LaTeX) e em PNG (visualização rápida / README).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela: roda em servidor/CI
import matplotlib.pyplot as plt  # noqa: E402

# Paleta categórica em ordem fixa (validada para daltonismo nas 3 primeiras posições).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#6250d6"
RED = "#e34948"

INK = "#0b0b0b"        # texto principal
INK_2 = "#52514e"      # texto secundário / eixos
GRID = "#e4e3df"       # grade discreta
SURFACE = "#fcfcfb"

# Papéis semânticos usados em todas as curvas de convergência.
C_BEST, C_MEAN, C_WORST = BLUE, ORANGE, AQUA

COL_W = 3.5    # largura de uma coluna IEEE (in)
PAGE_W = 7.16  # largura da página (duas colunas)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "font.size": 8.5,
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,  # fontes TrueType embutidas (aceitas pelo IEEE PDF eXpress)
    })


def save(fig, out_dir: Path, name: str) -> list[Path]:
    """Salva a figura em PDF e PNG e fecha o objeto para liberar memória."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.03)
        paths.append(p)
    plt.close(fig)
    return paths

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL = "#2a78d6"
LARANJA = "#eb6834"
VERDE_AGUA = "#1baf7a"
AMARELO = "#eda100"
VIOLETA = "#6250d6"
VERMELHO = "#e34948"

TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
GRADE = "#e4e3df"
SUPERFICIE = "#fcfcfb"

COR_MELHOR, COR_MEDIO, COR_PIOR = AZUL, LARANJA, VERDE_AGUA

LARG_COLUNA = 3.5
LARG_PAGINA = 7.16


def aplicar_estilo() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": TINTA_2,
        "axes.linewidth": 0.6,
        "axes.labelcolor": TINTA,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRADE,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": TINTA_2,
        "ytick.color": TINTA_2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "font.size": 8.5,
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
    })


def salvar_figura(fig, pasta: Path, nome: str) -> list[Path]:
    pasta.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for ext in ("pdf", "png"):
        caminho = pasta / f"{nome}.{ext}"
        fig.savefig(caminho, bbox_inches="tight", pad_inches=0.03)
        caminhos.append(caminho)
    plt.close(fig)
    return caminhos

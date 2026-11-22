"""Gráficos da Atividade 2 (superfície, população, convergência e distribuição)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, NullFormatter

from ..common import plotstyle as ps
from ..common.stats import history_band
from .rastrigin import LOW, UP, rastrigin

# Rampa sequencial de um único matiz (azul, claro -> escuro) para magnitude de f.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def _grid(n: int = 400):
    g = np.linspace(LOW, UP, n)
    X, Y = np.meshgrid(g, g)
    Z = rastrigin(np.stack([X, Y], axis=-1))
    return X, Y, Z


def plot_surface(best_points: np.ndarray, out_dir: Path, name: str) -> list[Path]:
    """Superfície 3D e curvas de nível de f (n=2), com os melhores de cada execução."""
    X, Y, Z = _grid(300)
    # Posições explícitas: tight_layout não lida bem com eixos 3D.
    fig = plt.figure(figsize=(ps.PAGE_W, 2.9))
    ax3 = fig.add_axes((0.0, 0.04, 0.46, 0.88), projection="3d")
    ax3.plot_surface(X, Y, Z, cmap=SEQ_BLUE, rstride=2, cstride=2, linewidth=0,
                     antialiased=True, rasterized=True)
    ax3.set_xlabel("$x_1$", labelpad=-4)
    ax3.set_ylabel("$x_2$", labelpad=-4)
    ax3.set_zlabel("$f(x)$", labelpad=-6)
    ax3.tick_params(labelsize=6.5, pad=-2)
    ax3.view_init(elev=32, azim=-60)
    ax3.set_title("Superfície de Rastrigin (n = 2)", y=1.0)

    ax = fig.add_axes((0.55, 0.25, 0.34, 0.66))
    cax = fig.add_axes((0.905, 0.25, 0.015, 0.66))
    cs = ax.contourf(X, Y, Z, levels=24, cmap=SEQ_BLUE)
    cs.set_rasterized(True)  # milhares de polígonos: rasteriza só o preenchimento
    cb = fig.colorbar(cs, cax=cax)
    cb.set_label("f(x)")
    cb.ax.tick_params(labelsize=6.5)
    ax.scatter(best_points[:, 0], best_points[:, 1], s=36, marker="o", facecolor=ps.ORANGE,
               edgecolor="white", lw=1.2, zorder=3, label=f"melhor de cada execução ({len(best_points)})")
    ax.scatter([0], [0], s=90, marker="+", color=ps.INK, lw=1.2, zorder=4, label="ótimo global (0, 0)")
    ax.set_xlabel("$x_1$", labelpad=1)
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title("Curvas de nível e soluções do AG")
    ax.legend(loc="upper center", bbox_to_anchor=(0.55, -0.16), ncol=2, fontsize=7)
    return ps.save(fig, out_dir, name)


def plot_population_snapshots(snaps: dict[int, np.ndarray], out_dir: Path, name: str) -> list[Path]:
    """Posição da população (n=2) em algumas gerações de uma execução."""
    X, Y, Z = _grid(300)
    gens = sorted(snaps)
    fig, axes = plt.subplots(1, len(gens), figsize=(ps.PAGE_W, 2.0), sharey=True)
    for ax, g in zip(np.atleast_1d(axes), gens):
        cs = ax.contourf(X, Y, Z, levels=24, cmap=SEQ_BLUE, alpha=0.85)
        cs.set_rasterized(True)
        P = snaps[g]
        ax.scatter(P[:, 0], P[:, 1], s=9, color=ps.ORANGE, edgecolor="white", lw=0.5, zorder=3)
        ax.scatter([0], [0], s=60, marker="+", color=ps.INK, lw=1.0, zorder=4)
        ax.set_title(f"geração {g}")
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("$x_1$")
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([-5, 0, 5])
    np.atleast_1d(axes)[0].set_ylabel("$x_2$")
    fig.tight_layout(w_pad=0.6)
    return ps.save(fig, out_dir, name)


def plot_convergence(results: dict[int, list], out_dir: Path, name: str) -> list[Path]:
    """Melhor/médio/pior f por geração (média das execuções), escala log."""
    dims = list(results)
    fig, axes = plt.subplots(1, len(dims), figsize=(ps.PAGE_W, 2.6))
    for j, (ax, n) in enumerate(zip(np.atleast_1d(axes), dims)):
        runs = results[n]
        gens = np.arange(len(runs[0].history["best"]))
        for metric, color, label in (("worst", ps.C_WORST, "pior"),
                                     ("mean", ps.C_MEAN, "médio"),
                                     ("best", ps.C_BEST, "melhor")):
            # média geométrica: adequada a valores que variam em ordens de grandeza
            logs = [np.log10(np.maximum(r.history[metric], 1e-12)) for r in runs]
            m, s = history_band(logs)
            ax.plot(gens, 10 ** m, color=color, lw=1.6, label=label)
            ax.fill_between(gens, 10 ** (m - s), 10 ** (m + s), color=color, alpha=0.10, lw=0)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"n = {n} ({len(runs)} execuções)")
        ax.set_xlabel("geração")
        if j == 0:
            ax.set_ylabel("f(x) (média geométrica, log)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles[::-1], [f"{l} da população" for l in labels[::-1]], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.07, 1, 1))
    return ps.save(fig, out_dir, name)


def plot_final_distribution(ga_best: dict[int, Sequence[float]], rs_best: dict[int, Sequence[float]],
                            out_dir: Path, name: str) -> list[Path]:
    dims = list(ga_best)
    fig, axes = plt.subplots(1, len(dims), figsize=(ps.COL_W, 2.3))
    rng = np.random.default_rng(0)
    for ax, n in zip(np.atleast_1d(axes), dims):
        data = [np.maximum(np.asarray(ga_best[n]), 1e-12), np.asarray(rs_best[n])]
        colors = [ps.BLUE, ps.ORANGE]
        bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False,
                        medianprops={"color": ps.INK, "lw": 1.2},
                        whiskerprops={"color": ps.INK_2, "lw": 0.8},
                        capprops={"color": ps.INK_2, "lw": 0.8})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.18)
            patch.set_edgecolor(c)
        for i, (d, c) in enumerate(zip(data, colors), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, d.size), d, s=10, color=c,
                       edgecolor="white", lw=0.6, zorder=3)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([1, 2], ["AG\n(DEAP)", "Busca\naleatória"])
        ax.set_title(f"n = {n}")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("melhor f final (log)")
    fig.tight_layout(w_pad=1.0)
    return ps.save(fig, out_dir, name)

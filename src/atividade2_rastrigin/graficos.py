from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, NullFormatter

from ..comum import estilo_graficos as eg
from ..comum.estatistica import faixa_historico
from .rastrigin import LIM_INF, LIM_SUP, rastrigin

AZUL_SEQUENCIAL = LinearSegmentedColormap.from_list(
    "azul_sequencial", ["#f3f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def _malha(n: int = 400):
    g = np.linspace(LIM_INF, LIM_SUP, n)
    X, Y = np.meshgrid(g, g)
    Z = rastrigin(np.stack([X, Y], axis=-1))
    return X, Y, Z


def grafico_superficie(melhores_pontos: np.ndarray, pasta: Path, nome: str) -> list[Path]:
    X, Y, Z = _malha(300)
    fig = plt.figure(figsize=(eg.LARG_PAGINA, 2.9))
    ax3 = fig.add_axes((0.0, 0.04, 0.46, 0.88), projection="3d")
    ax3.plot_surface(X, Y, Z, cmap=AZUL_SEQUENCIAL, rstride=2, cstride=2, linewidth=0,
                     antialiased=True, rasterized=True)
    ax3.set_xlabel("$x_1$", labelpad=-4)
    ax3.set_ylabel("$x_2$", labelpad=-4)
    ax3.set_zlabel("$f(x)$", labelpad=-6)
    ax3.tick_params(labelsize=6.5, pad=-2)
    ax3.view_init(elev=32, azim=-60)
    ax3.set_title("Superfície de Rastrigin (n = 2)", y=1.0)

    ax = fig.add_axes((0.55, 0.25, 0.34, 0.66))
    ax_barra = fig.add_axes((0.905, 0.25, 0.015, 0.66))
    curvas = ax.contourf(X, Y, Z, levels=24, cmap=AZUL_SEQUENCIAL)
    curvas.set_rasterized(True)
    barra = fig.colorbar(curvas, cax=ax_barra)
    barra.set_label("f(x)")
    barra.ax.tick_params(labelsize=6.5)
    ax.scatter(melhores_pontos[:, 0], melhores_pontos[:, 1], s=36, marker="o", facecolor=eg.LARANJA,
               edgecolor="white", lw=1.2, zorder=3,
               label=f"melhor de cada execução ({len(melhores_pontos)})")
    ax.scatter([0], [0], s=90, marker="+", color=eg.TINTA, lw=1.2, zorder=4, label="ótimo global (0, 0)")
    ax.set_xlabel("$x_1$", labelpad=1)
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title("Curvas de nível e soluções do AG")
    ax.legend(loc="upper center", bbox_to_anchor=(0.55, -0.16), ncol=2, fontsize=7)
    return eg.salvar_figura(fig, pasta, nome)


def grafico_populacao(retratos: dict[int, np.ndarray], pasta: Path, nome: str) -> list[Path]:
    X, Y, Z = _malha(300)
    geracoes = sorted(retratos)
    fig, eixos = plt.subplots(1, len(geracoes), figsize=(eg.LARG_PAGINA, 2.0), sharey=True)
    for ax, g in zip(np.atleast_1d(eixos), geracoes):
        curvas = ax.contourf(X, Y, Z, levels=24, cmap=AZUL_SEQUENCIAL, alpha=0.85)
        curvas.set_rasterized(True)
        P = retratos[g]
        ax.scatter(P[:, 0], P[:, 1], s=9, color=eg.LARANJA, edgecolor="white", lw=0.5, zorder=3)
        ax.scatter([0], [0], s=60, marker="+", color=eg.TINTA, lw=1.0, zorder=4)
        ax.set_title(f"geração {g}")
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("$x_1$")
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([-5, 0, 5])
    np.atleast_1d(eixos)[0].set_ylabel("$x_2$")
    fig.tight_layout(w_pad=0.6)
    return eg.salvar_figura(fig, pasta, nome)


def grafico_convergencia(resultados: dict[int, list], pasta: Path, nome: str) -> list[Path]:
    dimensoes = list(resultados)
    fig, eixos = plt.subplots(1, len(dimensoes), figsize=(eg.LARG_PAGINA, 2.6))
    for j, (ax, n) in enumerate(zip(np.atleast_1d(eixos), dimensoes)):
        execucoes = resultados[n]
        geracoes = np.arange(len(execucoes[0].historico["melhor"]))
        for metrica, cor, rotulo in (("pior", eg.COR_PIOR, "pior"),
                                     ("medio", eg.COR_MEDIO, "médio"),
                                     ("melhor", eg.COR_MELHOR, "melhor")):
            logs = [np.log10(np.maximum(r.historico[metrica], 1e-12)) for r in execucoes]
            m, s = faixa_historico(logs)
            ax.plot(geracoes, 10 ** m, color=cor, lw=1.6, label=rotulo)
            ax.fill_between(geracoes, 10 ** (m - s), 10 ** (m + s), color=cor, alpha=0.10, lw=0)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"n = {n} ({len(execucoes)} execuções)")
        ax.set_xlabel("geração")
        if j == 0:
            ax.set_ylabel("f(x) (média geométrica, log)")
    alcas, rotulos = np.atleast_1d(eixos)[0].get_legend_handles_labels()
    fig.legend(alcas[::-1], [f"{r} da população" for r in rotulos[::-1]], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.07, 1, 1))
    return eg.salvar_figura(fig, pasta, nome)


def grafico_distribuicao_final(melhores_ag: dict[int, Sequence[float]],
                               melhores_aleatoria: dict[int, Sequence[float]],
                               pasta: Path, nome: str) -> list[Path]:
    dimensoes = list(melhores_ag)
    fig, eixos = plt.subplots(1, len(dimensoes), figsize=(eg.LARG_COLUNA, 2.3))
    rng = np.random.default_rng(0)
    for ax, n in zip(np.atleast_1d(eixos), dimensoes):
        dados = [np.maximum(np.asarray(melhores_ag[n]), 1e-12), np.asarray(melhores_aleatoria[n])]
        cores = [eg.AZUL, eg.LARANJA]
        caixas = ax.boxplot(dados, widths=0.45, patch_artist=True, showfliers=False,
                            medianprops={"color": eg.TINTA, "lw": 1.2},
                            whiskerprops={"color": eg.TINTA_2, "lw": 0.8},
                            capprops={"color": eg.TINTA_2, "lw": 0.8})
        for caixa, c in zip(caixas["boxes"], cores):
            caixa.set_facecolor(c)
            caixa.set_alpha(0.18)
            caixa.set_edgecolor(c)
        for i, (d, c) in enumerate(zip(dados, cores), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, d.size), d, s=10, color=c,
                       edgecolor="white", lw=0.6, zorder=3)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([1, 2], ["AG\n(DEAP)", "Busca\naleatória"])
        ax.set_title(f"n = {n}")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(eixos)[0].set_ylabel("melhor f final (log)")
    fig.tight_layout(w_pad=1.0)
    return eg.salvar_figura(fig, pasta, nome)

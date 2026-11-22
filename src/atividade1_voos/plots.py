"""Gráficos da Atividade 1 (convergência, distribuição, trade-off e roteiro)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from ..common import plotstyle as ps
from ..common.stats import history_band
from .flights import min_to_hhmm
from .model import Scenario


def _hhmm_formatter():
    return FuncFormatter(lambda v, _: min_to_hhmm(v))


def _thousands(v: float, decimals: int = 0) -> str:
    """Separador de milhar no padrão brasileiro: 12345 -> '12.345'."""
    return f"{v:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


SHORT_TITLE = {"A": "Cenário A (dias distintos)", "B": "Cenário B (mesa-redonda 14:00-16:00)"}


def _end_label(ax, x, y, text, color=ps.INK_2, dy=0.0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points",
                va="center", ha="left", fontsize=7, color=color)


def plot_convergence(results: dict[str, list], optimum: dict[str, float],
                     scenarios: dict[str, Scenario], out_dir: Path, name: str) -> list[Path]:
    """Melhor/médio/pior por geração (média de todas as execuções) + diversidade."""
    keys = list(results)
    fig, axes = plt.subplots(2, len(keys), figsize=(ps.PAGE_W, 4.1), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    axes = np.atleast_2d(axes).reshape(2, len(keys))
    for j, k in enumerate(keys):
        runs = results[k]
        ax, axd = axes[0, j], axes[1, j]
        gens = np.arange(len(runs[0].history["best"]))
        for metric, color, label in (("worst", ps.C_WORST, "pior"),
                                     ("mean", ps.C_MEAN, "médio"),
                                     ("best", ps.C_BEST, "melhor")):
            m, s = history_band([r.history[metric] for r in runs])
            ax.plot(gens, m, color=color, lw=1.6, label=f"{label} (média de {len(runs)} exec.)")
            ax.fill_between(gens, np.maximum(m - s, 1), m + s, color=color, alpha=0.10, lw=0)
        ax.axhline(optimum[k], color=ps.INK, lw=0.9, ls=(0, (4, 3)))
        ax.annotate(f"ótimo exato F* = {_thousands(optimum[k])}", (gens[-1], optimum[k]),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                    fontsize=7, color=ps.INK)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _thousands(v)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(SHORT_TITLE.get(k, f"Cenário {k}"))
        if j == 0:
            ax.set_ylabel("fitness F (EUR, log)")
            ax.legend(loc="upper right")
        pop = runs[0].final_population.shape[0]
        m, s = history_band([r.history["unique"] / pop for r in runs])
        axd.plot(gens, m * 100, color=ps.VIOLET, lw=1.6)
        axd.fill_between(gens, (m - s) * 100, (m + s) * 100, color=ps.VIOLET, alpha=0.10, lw=0)
        axd.set_ylim(0, 105)
        axd.set_xlabel("geração")
        if j == 0:
            axd.set_ylabel("genótipos\núnicos (%)")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(h_pad=0.6, w_pad=1.2)
    return ps.save(fig, out_dir, name)


def plot_final_distribution(ga_best: dict[str, Sequence[float]], rs_best: dict[str, Sequence[float]],
                            optimum: dict[str, float], out_dir: Path, name: str) -> list[Path]:
    """Distribuição do melhor fitness final: AG x busca aleatória (mesmo orçamento)."""
    keys = list(ga_best)
    fig, axes = plt.subplots(1, len(keys), figsize=(ps.COL_W, 2.3))
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, k in zip(axes, keys):
        data = [np.asarray(ga_best[k]), np.asarray(rs_best[k])]
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
            x = i + rng.uniform(-0.12, 0.12, size=d.size)
            ax.scatter(x, d, s=10, color=c, edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(optimum[k], color=ps.INK, lw=0.9, ls=(0, (4, 3)))
        ax.set_xticks([1, 2], ["AG", "Busca\naleatória"])
        ax.set_title(f"Cenário {k}")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _thousands(v)))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("melhor F final (EUR)")
    fig.tight_layout(w_pad=1.0)
    return ps.save(fig, out_dir, name)


def _place_lambda_labels(ax, groups: dict[tuple[float, float], list[float]],
                         min_gap_pt: float = 9.0) -> None:
    """Rótulos de lambda à direita dos pontos, com espaçamento vertical mínimo.

    Percorre os pontos de baixo para cima; cada rótulo fica 8 pt acima do seu
    ponto, mas pelo menos ``min_gap_pt`` acima do rótulo anterior. Uma linha-guia
    fina liga cada rótulo ao seu ponto. Deve ser chamada depois do ``tight_layout``.
    """
    to_pt = 72.0 / ax.figure.dpi
    prev_y = None
    for (c, w), lams in sorted(groups.items(), key=lambda t: t[0][0]):
        _, y_disp = ax.transData.transform((w, c))
        y_pt = y_disp * to_pt
        target = y_pt + 8
        if prev_y is not None and target < prev_y + min_gap_pt:
            target = prev_y + min_gap_pt
        prev_y = target
        txt = "λ=" + ", ".join(f"{lam:g}" for lam in lams)
        ax.annotate(txt, (w, c), xytext=(14, target - y_pt), textcoords="offset points",
                    fontsize=6.5, color=ps.INK_2, va="center", ha="left",
                    arrowprops={"arrowstyle": "-", "color": ps.INK_2, "lw": 0.5,
                                "shrinkA": 1, "shrinkB": 5})


def plot_tradeoff(panels: list[dict], out_dir: Path, name: str) -> list[Path]:
    """Custo x espera: amostra aleatória, fronteira de Pareto exata e soluções do AG.

    Cada painel é um dict com: key, sample (M,2), front (K,2), ga (lista de
    (lambda, custo, espera)), exact (lista de (lambda, custo, espera)).
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(ps.PAGE_W, 2.75))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        s = p["sample"]
        ax.scatter(s[:, 1], s[:, 0], s=3, color="#b9b8b2", alpha=0.5, lw=0,
                   label="soluções factíveis aleatórias", rasterized=True)
        f = p["front"][np.argsort(p["front"][:, 1])]
        ax.plot(f[:, 1], f[:, 0], color=ps.INK_2, lw=1.0, drawstyle="steps-post",
                label="fronteira de Pareto exata")
        ex = np.asarray(p["exact"])
        ax.scatter(ex[:, 2], ex[:, 1], s=60, facecolor="none", edgecolor=ps.INK, lw=1.0,
                   zorder=4, label=r"ótimo exato de $F_\lambda$")
        ga = np.asarray(p["ga"])
        ax.scatter(ga[:, 2], ga[:, 1], s=16, color=ps.BLUE, edgecolor="white", lw=0.8,
                   zorder=5, label="melhor do AG (cada execução)")
        # agrupa valores de lambda que levam ao mesmo ótimo (evita rótulos sobrepostos)
        groups: dict[tuple[float, float], list[float]] = {}
        for lam, c, w in p["exact"]:
            groups.setdefault((c, w), []).append(lam)
        ax._lambda_groups = groups  # rótulos posicionados após o tight_layout
        ax.set_xlabel("espera total W (min)")
        ax.set_title(SHORT_TITLE.get(p["key"], f"Cenário {p['key']}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _thousands(v)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _thousands(v)))
    axes[0].set_ylabel("custo total C (EUR)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.01), markerscale=1.5)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.07, 1, 1))
    for ax in axes:
        _place_lambda_labels(ax, ax._lambda_groups)
    return ps.save(fig, out_dir, name)


def plot_schedule(sol: dict, sc: Scenario, out_dir: Path, name: str) -> list[Path]:
    """Roteiro (gráfico de Gantt) da melhor solução: voos, esperas e van."""
    rows = sol["rows"]
    n = len(rows)
    y = np.arange(n)[::-1]
    bar_h = 0.34

    if sc.same_day:
        fig, ax = plt.subplots(figsize=(ps.PAGE_W, 2.35))
        axes = [ax, ax]
        ax.axvspan(sc.event_start, sc.event_end, color="#f0efec", lw=0, zorder=0)
        ax.text((sc.event_start + sc.event_end) / 2, n - 0.45, "mesa-redonda",
                ha="center", va="center", fontsize=7, color=ps.INK_2)
        # translados da van
        for x0, x1 in ((sol["van_out"], sol["van_at_event"]),
                       (sc.event_end, sol["van_drop"])):
            ax.axvspan(x0, x1, color=ps.VIOLET, alpha=0.08, lw=0, zorder=0)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(ps.PAGE_W, 2.35), sharey=True)
        axes[0].set_title("Dia da chegada (ida)")
        axes[1].set_title("Dia do retorno (volta)")

    ax_out, ax_ret = axes
    for yi, r in zip(y, rows):
        # ida: voo + espera até a van sair
        ax_out.barh(yi, r["out_arr"] - r["out_dep"], left=r["out_dep"], height=bar_h,
                    color=ps.BLUE, zorder=3)
        if r["wait_out"] > 0:
            ax_out.barh(yi, r["wait_out"], left=r["out_arr"], height=bar_h,
                        color=ps.ORANGE, alpha=0.55, zorder=3)
        ax_out.text(r["out_dep"] - 6, yi, f"€{r['out_price']}", ha="right", va="center",
                    fontsize=6.5, color=ps.INK_2)
        # volta: espera desde a van até a partida + voo
        if r["wait_ret"] > 0:
            ax_ret.barh(yi, r["wait_ret"], left=sol["van_drop"], height=bar_h,
                        color=ps.ORANGE, alpha=0.55, zorder=3)
        ax_ret.barh(yi, r["ret_arr"] - r["ret_dep"], left=r["ret_dep"], height=bar_h,
                    color=ps.BLUE, zorder=3)
        ax_ret.text(r["ret_arr"] + 6, yi, f"€{r['ret_price']}", ha="left", va="center",
                    fontsize=6.5, color=ps.INK_2)

    # rótulos da van no topo do painel: "sai" à esquerda da linha, "chega" à direita
    for ax, x, lbl, ha, dx in ((ax_out, sol["van_out"], "van sai de FCO", "right", -4),
                               (ax_ret, sol["van_drop"], "van chega a FCO", "left", 4)):
        ax.axvline(x, color=ps.INK, lw=0.9, ls=(0, (4, 3)), zorder=4)
        ax.annotate(f"{lbl} {min_to_hhmm(x)}", (x, n - 0.45), xytext=(dx, 0),
                    textcoords="offset points", ha=ha, va="center", fontsize=6.8, color=ps.INK)

    def _ticks(lo, hi):
        step = 60 if hi - lo <= 6 * 60 else 120
        return np.arange((lo // step + 1) * step, hi, step)

    for ax in set(axes):
        ax.set_yticks(y, [r["origin"] for r in rows])
        ax.xaxis.set_major_formatter(_hhmm_formatter())
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.55, n + 0.05)
    if not sc.same_day:
        lo = min(min(r["out_dep"] for r in rows) - 75, sol["van_out"] - 150)
        hi = max(sol["van_out"], max(r["out_arr"] for r in rows)) + 20
        ax_out.set_xlim(lo, hi)
        ax_out.set_xticks(_ticks(lo, hi))
        lo2 = sol["van_drop"] - 20
        hi2 = max(max(r["ret_arr"] for r in rows) + 75, sol["van_drop"] + 150)
        ax_ret.set_xlim(lo2, hi2)
        ax_ret.set_xticks(_ticks(lo2, hi2))
    else:
        ax_out.set_xlim(5 * 60, 24 * 60 + 15)
        ax_out.set_xticks(np.arange(6 * 60, 25 * 60, 120))

    # legenda manual (identidade não depende só da cor: rótulos explícitos)
    from matplotlib.patches import Patch
    handles = [Patch(color=ps.BLUE, label="voo"),
               Patch(color=ps.ORANGE, alpha=0.55, label="espera no aeroporto")]
    if sc.same_day:
        handles.append(Patch(color=ps.VIOLET, alpha=0.15, label="translado de van"))
    axes[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
    fig.suptitle(
        f"Cenário {sc.key}: custo EUR {_thousands(sol['cost'])}, espera {_thousands(sol['wait'])} min, "
        f"F = {_thousands(sol['fitness'])}", fontsize=8.5, y=1.0)
    fig.tight_layout()
    return ps.save(fig, out_dir, name)

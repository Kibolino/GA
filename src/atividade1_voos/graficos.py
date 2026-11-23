from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from ..comum import estilo_graficos as eg
from ..comum.estatistica import faixa_historico
from .modelo import Cenario
from .voos import min_para_hhmm

TITULO_CURTO = {"A": "Cenário A (dias distintos)", "B": "Cenário B (mesa-redonda 14:00-16:00)"}


def _formatador_hhmm():
    return FuncFormatter(lambda v, _: min_para_hhmm(v))


def _milhar(v: float, casas: int = 0) -> str:
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def grafico_convergencia(resultados: dict[str, list], otimo: dict[str, float],
                         cenarios: dict[str, Cenario], pasta: Path, nome: str) -> list[Path]:
    chaves = list(resultados)
    fig, eixos = plt.subplots(2, len(chaves), figsize=(eg.LARG_PAGINA, 4.1), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1]})
    eixos = np.atleast_2d(eixos).reshape(2, len(chaves))
    for j, k in enumerate(chaves):
        execucoes = resultados[k]
        ax, ax_div = eixos[0, j], eixos[1, j]
        geracoes = np.arange(len(execucoes[0].historico["melhor"]))
        for metrica, cor, rotulo in (("pior", eg.COR_PIOR, "pior"),
                                     ("medio", eg.COR_MEDIO, "médio"),
                                     ("melhor", eg.COR_MELHOR, "melhor")):
            m, s = faixa_historico([r.historico[metrica] for r in execucoes])
            ax.plot(geracoes, m, color=cor, lw=1.6, label=f"{rotulo} (média de {len(execucoes)} exec.)")
            ax.fill_between(geracoes, np.maximum(m - s, 1), m + s, color=cor, alpha=0.10, lw=0)
        ax.axhline(otimo[k], color=eg.TINTA, lw=0.9, ls=(0, (4, 3)))
        ax.annotate(f"ótimo exato F* = {_milhar(otimo[k])}", (geracoes[-1], otimo[k]),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                    fontsize=7, color=eg.TINTA)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _milhar(v)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(TITULO_CURTO.get(k, f"Cenário {k}"))
        if j == 0:
            ax.set_ylabel("fitness F (EUR, log)")
            ax.legend(loc="upper right")
        tam_pop = execucoes[0].populacao_final.shape[0]
        m, s = faixa_historico([r.historico["unicos"] / tam_pop for r in execucoes])
        ax_div.plot(geracoes, m * 100, color=eg.VIOLETA, lw=1.6)
        ax_div.fill_between(geracoes, (m - s) * 100, (m + s) * 100, color=eg.VIOLETA, alpha=0.10, lw=0)
        ax_div.set_ylim(0, 105)
        ax_div.set_xlabel("geração")
        if j == 0:
            ax_div.set_ylabel("genótipos\núnicos (%)")
    fig.align_ylabels(eixos[:, 0])
    fig.tight_layout(h_pad=0.6, w_pad=1.2)
    return eg.salvar_figura(fig, pasta, nome)


def grafico_distribuicao_final(melhores_ag: dict[str, Sequence[float]],
                               melhores_aleatoria: dict[str, Sequence[float]],
                               otimo: dict[str, float], pasta: Path, nome: str) -> list[Path]:
    chaves = list(melhores_ag)
    fig, eixos = plt.subplots(1, len(chaves), figsize=(eg.LARG_COLUNA, 2.3))
    eixos = np.atleast_1d(eixos)
    rng = np.random.default_rng(0)
    for ax, k in zip(eixos, chaves):
        dados = [np.asarray(melhores_ag[k]), np.asarray(melhores_aleatoria[k])]
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
            x = i + rng.uniform(-0.12, 0.12, size=d.size)
            ax.scatter(x, d, s=10, color=c, edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(otimo[k], color=eg.TINTA, lw=0.9, ls=(0, (4, 3)))
        ax.set_xticks([1, 2], ["AG", "Busca\naleatória"])
        ax.set_title(f"Cenário {k}")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _milhar(v)))
        ax.grid(axis="x", visible=False)
    eixos[0].set_ylabel("melhor F final (EUR)")
    fig.tight_layout(w_pad=1.0)
    return eg.salvar_figura(fig, pasta, nome)


def _posicionar_rotulos_lambda(ax, grupos: dict[tuple[float, float], list[float]],
                               espaco_min_pt: float = 9.0) -> None:
    para_pt = 72.0 / ax.figure.dpi
    y_anterior = None
    for (c, w), lambdas in sorted(grupos.items(), key=lambda t: t[0][0]):
        _, y_tela = ax.transData.transform((w, c))
        y_pt = y_tela * para_pt
        alvo = y_pt + 8
        if y_anterior is not None and alvo < y_anterior + espaco_min_pt:
            alvo = y_anterior + espaco_min_pt
        y_anterior = alvo
        texto = "λ=" + ", ".join(f"{lam:g}" for lam in lambdas)
        ax.annotate(texto, (w, c), xytext=(14, alvo - y_pt), textcoords="offset points",
                    fontsize=6.5, color=eg.TINTA_2, va="center", ha="left",
                    arrowprops={"arrowstyle": "-", "color": eg.TINTA_2, "lw": 0.5,
                                "shrinkA": 1, "shrinkB": 5})


def grafico_custo_espera(paineis: list[dict], pasta: Path, nome: str) -> list[Path]:
    fig, eixos = plt.subplots(1, len(paineis), figsize=(eg.LARG_PAGINA, 2.75))
    eixos = np.atleast_1d(eixos)
    grupos_por_eixo = []
    for ax, p in zip(eixos, paineis):
        s = p["amostra"]
        ax.scatter(s[:, 1], s[:, 0], s=3, color="#b9b8b2", alpha=0.5, lw=0,
                   label="soluções factíveis aleatórias", rasterized=True)
        f = p["fronteira"][np.argsort(p["fronteira"][:, 1])]
        ax.plot(f[:, 1], f[:, 0], color=eg.TINTA_2, lw=1.0, drawstyle="steps-post",
                label="fronteira de Pareto exata")
        ex = np.asarray(p["exato"])
        ax.scatter(ex[:, 2], ex[:, 1], s=60, facecolor="none", edgecolor=eg.TINTA, lw=1.0,
                   zorder=4, label=r"ótimo exato de $F_\lambda$")
        ag = np.asarray(p["ag"])
        ax.scatter(ag[:, 2], ag[:, 1], s=16, color=eg.AZUL, edgecolor="white", lw=0.8,
                   zorder=5, label="melhor do AG (cada execução)")
        grupos: dict[tuple[float, float], list[float]] = {}
        for lam, c, w in p["exato"]:
            grupos.setdefault((c, w), []).append(lam)
        grupos_por_eixo.append(grupos)
        ax.set_xlabel("espera total W (min)")
        ax.set_title(TITULO_CURTO.get(p["chave"], f"Cenário {p['chave']}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _milhar(v)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _milhar(v)))
    eixos[0].set_ylabel("custo total C (EUR)")
    alcas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(alcas, rotulos, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.01), markerscale=1.5)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.07, 1, 1))
    for ax, grupos in zip(eixos, grupos_por_eixo):
        _posicionar_rotulos_lambda(ax, grupos)
    return eg.salvar_figura(fig, pasta, nome)


def grafico_roteiro(sol: dict, cen: Cenario, pasta: Path, nome: str) -> list[Path]:
    linhas = sol["linhas"]
    n = len(linhas)
    y = np.arange(n)[::-1]
    altura = 0.34

    if cen.mesmo_dia:
        fig, ax = plt.subplots(figsize=(eg.LARG_PAGINA, 2.35))
        eixos = [ax, ax]
        ax.axvspan(cen.inicio_evento, cen.fim_evento, color="#f0efec", lw=0, zorder=0)
        ax.text((cen.inicio_evento + cen.fim_evento) / 2, n - 0.45, "mesa-redonda",
                ha="center", va="center", fontsize=7, color=eg.TINTA_2)
        for x0, x1 in ((sol["van_saida"], sol["van_no_evento"]),
                       (cen.fim_evento, sol["van_entrega"])):
            ax.axvspan(x0, x1, color=eg.VIOLETA, alpha=0.08, lw=0, zorder=0)
    else:
        fig, eixos = plt.subplots(1, 2, figsize=(eg.LARG_PAGINA, 2.35), sharey=True)
        eixos[0].set_title("Dia da chegada (ida)")
        eixos[1].set_title("Dia do retorno (volta)")

    ax_ida, ax_volta = eixos
    for yi, r in zip(y, linhas):
        ax_ida.barh(yi, r["ida_chegada"] - r["ida_partida"], left=r["ida_partida"], height=altura,
                    color=eg.AZUL, zorder=3)
        if r["espera_ida"] > 0:
            ax_ida.barh(yi, r["espera_ida"], left=r["ida_chegada"], height=altura,
                        color=eg.LARANJA, alpha=0.55, zorder=3)
        ax_ida.text(r["ida_partida"] - 6, yi, f"€{r['ida_preco']}", ha="right", va="center",
                    fontsize=6.5, color=eg.TINTA_2)
        if r["espera_volta"] > 0:
            ax_volta.barh(yi, r["espera_volta"], left=sol["van_entrega"], height=altura,
                          color=eg.LARANJA, alpha=0.55, zorder=3)
        ax_volta.barh(yi, r["volta_chegada"] - r["volta_partida"], left=r["volta_partida"],
                      height=altura, color=eg.AZUL, zorder=3)
        ax_volta.text(r["volta_chegada"] + 6, yi, f"€{r['volta_preco']}", ha="left", va="center",
                      fontsize=6.5, color=eg.TINTA_2)

    for ax, x, rotulo, alinhamento, dx in (
            (ax_ida, sol["van_saida"], "van sai de FCO", "right", -4),
            (ax_volta, sol["van_entrega"], "van chega a FCO", "left", 4)):
        ax.axvline(x, color=eg.TINTA, lw=0.9, ls=(0, (4, 3)), zorder=4)
        ax.annotate(f"{rotulo} {min_para_hhmm(x)}", (x, n - 0.45), xytext=(dx, 0),
                    textcoords="offset points", ha=alinhamento, va="center", fontsize=6.8,
                    color=eg.TINTA)

    def _marcas(inicio, fim):
        passo = 60 if fim - inicio <= 6 * 60 else 120
        return np.arange((inicio // passo + 1) * passo, fim, passo)

    for ax in set(eixos):
        ax.set_yticks(y, [r["origem"] for r in linhas])
        ax.xaxis.set_major_formatter(_formatador_hhmm())
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.55, n + 0.05)
    if not cen.mesmo_dia:
        ini = min(min(r["ida_partida"] for r in linhas) - 75, sol["van_saida"] - 150)
        fim = max(sol["van_saida"], max(r["ida_chegada"] for r in linhas)) + 20
        ax_ida.set_xlim(ini, fim)
        ax_ida.set_xticks(_marcas(ini, fim))
        ini2 = sol["van_entrega"] - 20
        fim2 = max(max(r["volta_chegada"] for r in linhas) + 75, sol["van_entrega"] + 150)
        ax_volta.set_xlim(ini2, fim2)
        ax_volta.set_xticks(_marcas(ini2, fim2))
    else:
        ax_ida.set_xlim(5 * 60, 24 * 60 + 15)
        ax_ida.set_xticks(np.arange(6 * 60, 25 * 60, 120))

    legenda = [Patch(color=eg.AZUL, label="voo"),
               Patch(color=eg.LARANJA, alpha=0.55, label="espera no aeroporto")]
    if cen.mesmo_dia:
        legenda.append(Patch(color=eg.VIOLETA, alpha=0.15, label="translado de van"))
    eixos[-1].legend(handles=legenda, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
    fig.suptitle(
        f"Cenário {cen.chave}: custo EUR {_milhar(sol['custo'])}, espera {_milhar(sol['espera'])} min, "
        f"F = {_milhar(sol['fitness'])}", fontsize=8.5, y=1.0)
    fig.tight_layout()
    return eg.salvar_figura(fig, pasta, nome)

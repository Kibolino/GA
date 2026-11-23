from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modelo import AvaliacaoSentido, Cenario, avaliar_ida, avaliar_volta
from .voos import TabelaVoos


def todas_combinacoes(limites: np.ndarray) -> np.ndarray:
    grades = np.indices(tuple(int(u) for u in limites), dtype=np.int16)
    return grades.reshape(len(limites), -1).T


@dataclass
class EnumeracaoSentido:
    combinacoes: np.ndarray
    aval: AvaliacaoSentido

    def fitness(self, cen: Cenario) -> np.ndarray:
        return self.aval.fitness(cen)


def enumerar_sentidos(tabela: TabelaVoos, cen: Cenario) -> tuple[EnumeracaoSentido, EnumeracaoSentido]:
    limite = tabela.limite_genes
    comb_ida = todas_combinacoes(limite[0::2]).astype(np.int64)
    comb_volta = todas_combinacoes(limite[1::2]).astype(np.int64)
    return (EnumeracaoSentido(comb_ida, avaliar_ida(tabela, cen, comb_ida)),
            EnumeracaoSentido(comb_volta, avaliar_volta(tabela, cen, comb_volta)))


@dataclass
class ResultadoExato:
    fitness: float
    individuo: np.ndarray
    n_otimos: int
    fitness_ida: float
    fitness_volta: float


def _intercalar(indices_ida: np.ndarray, indices_volta: np.ndarray) -> np.ndarray:
    individuo = np.empty(indices_ida.size * 2, dtype=np.int64)
    individuo[0::2], individuo[1::2] = indices_ida, indices_volta
    return individuo


def otimo_exato(tabela: TabelaVoos, cen: Cenario,
                sentidos: tuple[EnumeracaoSentido, EnumeracaoSentido] | None = None,
                tol: float = 1e-9) -> ResultadoExato:
    ida, volta = sentidos or enumerar_sentidos(tabela, cen)
    fi, fv = ida.fitness(cen), volta.fitness(cen)
    ii, iv = int(np.argmin(fi)), int(np.argmin(fv))
    n_otimos = int(np.sum(fi <= fi[ii] + tol)) * int(np.sum(fv <= fv[iv] + tol))
    return ResultadoExato(
        fitness=float(fi[ii] + fv[iv]),
        individuo=_intercalar(ida.combinacoes[ii], volta.combinacoes[iv]),
        n_otimos=n_otimos,
        fitness_ida=float(fi[ii]),
        fitness_volta=float(fv[iv]),
    )


def mascara_pareto(custo: np.ndarray, espera: np.ndarray) -> np.ndarray:
    ordem = np.lexsort((espera, custo))
    w = espera[ordem]
    menor_anterior = np.concatenate([[np.inf], np.minimum.accumulate(w)[:-1]])
    manter = np.zeros(custo.size, dtype=bool)
    manter[ordem[w < menor_anterior]] = True
    return manter


def fronteira_pareto(tabela: TabelaVoos, cen: Cenario,
                     sentidos: tuple[EnumeracaoSentido, EnumeracaoSentido] | None = None) -> np.ndarray:
    ida, volta = sentidos or enumerar_sentidos(tabela, cen)
    fronteiras = []
    for sentido in (ida, volta):
        factivel = sentido.aval.n_violacoes == 0
        c = sentido.aval.custo[factivel].astype(float)
        w = sentido.aval.espera[factivel].astype(float)
        pontos = np.unique(np.column_stack([c, w]), axis=0)
        fronteiras.append(pontos[mascara_pareto(pontos[:, 0], pontos[:, 1])])
    fi, fv = fronteiras
    soma = np.unique((fi[:, None, :] + fv[None, :, :]).reshape(-1, 2), axis=0)
    fronteira = soma[mascara_pareto(soma[:, 0], soma[:, 1])]
    return fronteira[np.argsort(fronteira[:, 0])]

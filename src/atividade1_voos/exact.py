"""Ótimo global exato por enumeração decomposta (usado só para *validar* o AG).

O espaço completo tem 10^12 cromossomos, inviável de enumerar. Porém a fitness
é separável em ida e volta (``model.py``): F(x) = F_ida(x_ida) + F_volta(x_volta).
Assim basta enumerar 10^6 combinações de ida e 10^6 de volta (vetorizado com
numpy, ~1 s) e somar os mínimos. Isso fornece:

* o valor ótimo F* e o(s) cromossomo(s) ótimo(s) de cada cenário;
* a fronteira de Pareto exata (custo x espera), obtida combinando as
  fronteiras de cada sentido (soma de Minkowski seguida de filtragem).

O AG não usa nada deste módulo durante a busca.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flights import FlightTable
from .model import Scenario, SideEval, evaluate_outbound, evaluate_return


def all_combinations(upper: np.ndarray) -> np.ndarray:
    """Todas as combinações de índices (produto cartesiano) como matriz (M, len(upper))."""
    grids = np.indices(tuple(int(u) for u in upper), dtype=np.int16)
    return grids.reshape(len(upper), -1).T


@dataclass
class SideEnumeration:
    combos: np.ndarray     # (M, n_pessoas)
    ev: SideEval

    def fitness(self, sc: Scenario) -> np.ndarray:
        return self.ev.fitness(sc)


def enumerate_sides(table: FlightTable, sc: Scenario) -> tuple[SideEnumeration, SideEnumeration]:
    up = table.gene_upper
    out_c = all_combinations(up[0::2]).astype(np.int64)
    ret_c = all_combinations(up[1::2]).astype(np.int64)
    return (SideEnumeration(out_c, evaluate_outbound(table, sc, out_c)),
            SideEnumeration(ret_c, evaluate_return(table, sc, ret_c)))


@dataclass
class ExactResult:
    fitness: float
    individual: np.ndarray
    n_optima: int          # nº de cromossomos distintos que atingem F*
    fitness_out: float
    fitness_ret: float


def _interleave(out_idx: np.ndarray, ret_idx: np.ndarray) -> np.ndarray:
    ind = np.empty(out_idx.size * 2, dtype=np.int64)
    ind[0::2], ind[1::2] = out_idx, ret_idx
    return ind


def exact_optimum(table: FlightTable, sc: Scenario,
                  sides: tuple[SideEnumeration, SideEnumeration] | None = None,
                  tol: float = 1e-9) -> ExactResult:
    out, ret = sides or enumerate_sides(table, sc)
    fo, fr = out.fitness(sc), ret.fitness(sc)
    io, ir = int(np.argmin(fo)), int(np.argmin(fr))
    n_opt = int(np.sum(fo <= fo[io] + tol)) * int(np.sum(fr <= fr[ir] + tol))
    return ExactResult(
        fitness=float(fo[io] + fr[ir]),
        individual=_interleave(out.combos[io], ret.combos[ir]),
        n_optima=n_opt,
        fitness_out=float(fo[io]),
        fitness_ret=float(fr[ir]),
    )


def pareto_mask(cost: np.ndarray, wait: np.ndarray) -> np.ndarray:
    """Máscara dos pontos não dominados (minimização nos dois eixos), O(M log M)."""
    # Ordena por custo crescente (desempate por espera). Um ponto é não dominado
    # se sua espera é estritamente menor que a de todos os pontos anteriores.
    order = np.lexsort((wait, cost))
    w = wait[order]
    prev_min = np.concatenate([[np.inf], np.minimum.accumulate(w)[:-1]])
    keep = np.zeros(cost.size, dtype=bool)
    keep[order[w < prev_min]] = True
    return keep


def pareto_front(table: FlightTable, sc: Scenario,
                 sides: tuple[SideEnumeration, SideEnumeration] | None = None) -> np.ndarray:
    """Fronteira de Pareto exata (custo, espera) considerando só soluções factíveis.

    Devolve matriz (K, 2) ordenada por custo crescente.
    """
    out, ret = sides or enumerate_sides(table, sc)
    fronts = []
    for side in (out, ret):
        feas = side.ev.n_viol == 0
        c, w = side.ev.cost[feas].astype(float), side.ev.wait[feas].astype(float)
        # remove duplicatas antes do filtro para acelerar
        pts = np.unique(np.column_stack([c, w]), axis=0)
        m = pareto_mask(pts[:, 0], pts[:, 1])
        fronts.append(pts[m])
    fo, fr = fronts
    comb = (fo[:, None, :] + fr[None, :, :]).reshape(-1, 2)
    comb = np.unique(comb, axis=0)
    m = pareto_mask(comb[:, 0], comb[:, 1])
    front = comb[m]
    return front[np.argsort(front[:, 0])]

"""Algoritmo Genético implementado do zero (apenas numpy) para cromossomos inteiros.

O motor é genérico: recebe o vetor ``gene_upper`` (nº de alelos de cada gene)
e uma função de fitness vetorizada ``pop (N x L) -> fitness (N,)`` a ser
*minimizada*. Todas as operações da população são feitas com numpy, sem laços
Python sobre indivíduos, o que torna viável rodar centenas de execuções.

Ciclo de uma geração
--------------------
1. Elitismo: os ``elitism`` melhores indivíduos são copiados sem alteração
   (e sem reavaliação) para a próxima geração.
2. Seleção: ``N - elitism`` pais são escolhidos por torneio de tamanho ``k``
   (amostragem com reposição; vence o de menor fitness).
3. Cruzamento: pares de pais consecutivos geram dois filhos com probabilidade
   ``crossover_prob``; caso contrário os filhos são cópias dos pais.
4. Mutação: cada gene de cada filho sofre mutação com probabilidade
   ``mutation_prob``; a mutação é "creep" (voo vizinho, +/-1) com probabilidade
   ``creep_prob`` ou "reset" (outro alelo sorteado uniformemente).
5. Mutação de bloco (opcional, ``shift_prob``): desloca todos os genes de um
   grupo (p.ex. todas as idas) pelo mesmo passo +/-1.
6. Avaliação dos filhos e substituição geracional.

Validade: cruzamento só troca genes *da mesma posição* e a mutação só produz
valores em ``[0, gene_upper)``; portanto nenhum operador cria voos inexistentes.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

FitnessFn = Callable[[np.ndarray], np.ndarray]
CROSSOVERS = ("uniform", "one_point", "two_point")


@dataclass(frozen=True)
class GAParams:
    pop_size: int = 100
    generations: int = 150
    tournament_size: int = 3
    crossover: str = "uniform"
    crossover_prob: float = 0.9
    mutation_prob: float = 1 / 12      # por gene (~1 gene mutado por filho)
    creep_prob: float = 0.5            # fração das mutações do tipo +/-1
    shift_prob: float = 0.0            # prob. (por filho) da mutação de deslocamento de bloco
    elitism: int = 2

    def __post_init__(self):
        if self.crossover not in CROSSOVERS:
            raise ValueError(f"crossover deve ser um de {CROSSOVERS}")
        if not 0 <= self.elitism < self.pop_size:
            raise ValueError("elitism deve estar em [0, pop_size)")
        if self.tournament_size < 1:
            raise ValueError("tournament_size >= 1")
        for name in ("crossover_prob", "mutation_prob", "creep_prob", "shift_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} deve estar em [0, 1]")

    @property
    def evaluations(self) -> int:
        """Nº de avaliações da fitness por execução (elites não são reavaliados)."""
        return self.pop_size + self.generations * (self.pop_size - self.elitism)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GAResult:
    best_individual: np.ndarray
    best_fitness: float
    history: dict[str, np.ndarray]       # best/mean/worst/std/unique por geração
    final_population: np.ndarray
    final_fitness: np.ndarray
    evaluations: int
    elapsed_s: float
    seed: int | None = None
    extra: dict = field(default_factory=dict)

    def first_hit(self, target: float, tol: float = 1e-9) -> int | None:
        """Primeira geração em que o melhor atingiu ``target`` (ou None)."""
        hits = np.flatnonzero(self.history["best"] <= target + tol)
        return int(hits[0]) if hits.size else None


# --------------------------------------------------------------------------
# Operadores (todos vetorizados; ``rng`` é um numpy.random.Generator)
# --------------------------------------------------------------------------

def init_population(rng: np.random.Generator, n: int, gene_upper: np.ndarray) -> np.ndarray:
    """População inicial uniforme: gene g ~ U{0, ..., gene_upper[g]-1}."""
    return rng.integers(0, gene_upper[None, :], size=(n, gene_upper.size))


def tournament_select(rng: np.random.Generator, fitness: np.ndarray, n_select: int,
                      k: int) -> np.ndarray:
    """Torneio determinístico de tamanho k (minimização). Retorna índices dos vencedores."""
    cand = rng.integers(0, fitness.size, size=(n_select, k))
    winners = np.argmin(fitness[cand], axis=1)
    return cand[np.arange(n_select), winners]


def crossover(rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray, kind: str,
              prob: float) -> tuple[np.ndarray, np.ndarray]:
    """Gera dois filhos por par de pais. Troca apenas genes de mesma posição."""
    m, L = p1.shape
    if kind == "uniform":
        mask = rng.random((m, L)) < 0.5
    elif kind == "one_point":
        cut = rng.integers(1, L, size=m)                       # corte entre genes
        mask = np.arange(L)[None, :] >= cut[:, None]
    elif kind == "two_point":
        a = rng.integers(1, L, size=m)
        b = rng.integers(1, L, size=m)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        pos = np.arange(L)[None, :]
        mask = (pos >= lo[:, None]) & (pos < hi[:, None])
    else:
        raise ValueError(kind)
    mask &= (rng.random(m) < prob)[:, None]                    # pares que não cruzam
    c1 = np.where(mask, p2, p1)
    c2 = np.where(mask, p1, p2)
    return c1, c2


def mutate(rng: np.random.Generator, pop: np.ndarray, gene_upper: np.ndarray,
           prob: float, creep_prob: float) -> np.ndarray:
    """Mutação por gene: creep (+/-1, refletido nas bordas) ou reset uniforme.

    O reset sorteia um alelo *diferente* do atual (``r`` em [0, U-2] e
    ``r += r >= x``), garantindo que toda mutação altere o voo.
    """
    out = pop.copy()
    hit = rng.random(pop.shape) < prob
    if not hit.any():
        return out
    up = np.broadcast_to(gene_upper, pop.shape)
    creep = rng.random(pop.shape) < creep_prob

    step = np.where(rng.random(pop.shape) < 0.5, -1, 1)
    cand = pop + step
    cand = np.where((cand < 0) | (cand >= up), pop - step, cand)   # reflexão na borda

    r = rng.integers(0, np.maximum(up - 1, 1))
    r = r + (r >= pop)
    reset = np.where(up > 1, r, pop)

    new = np.where(creep, cand, reset)
    out[hit] = new[hit]
    return out


def block_shift(rng: np.random.Generator, pop: np.ndarray, gene_upper: np.ndarray,
                prob: float, groups: tuple[np.ndarray, ...]) -> np.ndarray:
    """Mutação de deslocamento de bloco (operador específico do domínio).

    Com probabilidade ``prob`` por indivíduo, sorteia um grupo de genes
    (p.ex. todos os voos de ida) e soma o mesmo passo +/-1 a todos eles,
    saturando nas bordas. Como os voos de cada rota estão ordenados por
    horário, isso desloca o grupo inteiro para o voo anterior/seguinte,
    mantendo os horários próximos entre si. A mutação gene a gene não
    consegue esse movimento: alterar um único voo de um grupo sincronizado
    quase sempre aumenta a espera (a van espera pelo último a chegar).
    """
    out = pop.copy()
    if prob <= 0 or not groups:
        return out
    who = np.flatnonzero(rng.random(pop.shape[0]) < prob)
    if who.size == 0:
        return out
    g_choice = rng.integers(0, len(groups), size=who.size)
    steps = np.where(rng.random(who.size) < 0.5, -1, 1)
    for gi, grp in enumerate(groups):
        rows = who[g_choice == gi]
        if rows.size == 0:
            continue
        st = steps[g_choice == gi][:, None]
        block = out[np.ix_(rows, grp)] + st
        out[np.ix_(rows, grp)] = np.clip(block, 0, gene_upper[grp] - 1)
    return out


# --------------------------------------------------------------------------
# Laço principal
# --------------------------------------------------------------------------

def run_ga(fitness_fn: FitnessFn, gene_upper: np.ndarray, params: GAParams,
           seed: int | None = None, shift_groups: tuple[np.ndarray, ...] = (),
           callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
           ) -> GAResult:
    """Executa uma rodada completa do AG e devolve o histórico de convergência.

    ``shift_groups``: grupos de genes usados pela mutação de bloco (vazio
    desativa o operador, que também é desligado com ``shift_prob=0``).
    """
    rng = np.random.default_rng(seed)
    gene_upper = np.asarray(gene_upper, dtype=np.int64)
    N, e = params.pop_size, params.elitism
    n_off = N - e
    n_pairs = (n_off + 1) // 2

    t0 = time.perf_counter()
    pop = init_population(rng, N, gene_upper)
    fit = fitness_fn(pop)
    evals = N

    G = params.generations
    hist = {k: np.empty(G + 1) for k in ("best", "mean", "worst", "std", "unique")}

    def record(g: int) -> None:
        hist["best"][g] = fit.min()
        hist["mean"][g] = fit.mean()
        hist["worst"][g] = fit.max()
        hist["std"][g] = fit.std()
        hist["unique"][g] = np.unique(pop, axis=0).shape[0]   # diversidade genotípica

    best_ind = pop[np.argmin(fit)].copy()
    best_fit = float(fit.min())
    record(0)
    if callback:
        callback(0, pop, fit)

    for g in range(1, G + 1):
        order = np.argsort(fit, kind="stable")
        elite_pop, elite_fit = pop[order[:e]], fit[order[:e]]

        parents = tournament_select(rng, fit, 2 * n_pairs, params.tournament_size)
        c1, c2 = crossover(rng, pop[parents[0::2]], pop[parents[1::2]],
                           params.crossover, params.crossover_prob)
        children = np.vstack([c1, c2])[:n_off]
        children = mutate(rng, children, gene_upper, params.mutation_prob, params.creep_prob)
        children = block_shift(rng, children, gene_upper, params.shift_prob, shift_groups)
        child_fit = fitness_fn(children)
        evals += children.shape[0]

        pop = np.vstack([elite_pop, children])
        fit = np.concatenate([elite_fit, child_fit])

        i = int(np.argmin(fit))
        if fit[i] < best_fit:                     # melhor global (relevante se elitism=0)
            best_fit, best_ind = float(fit[i]), pop[i].copy()
        record(g)
        if callback:
            callback(g, pop, fit)

    return GAResult(
        best_individual=best_ind,
        best_fitness=best_fit,
        history=hist,
        final_population=pop,
        final_fitness=fit,
        evaluations=evals,
        elapsed_s=time.perf_counter() - t0,
        seed=seed,
    )


def random_search(fitness_fn: FitnessFn, gene_upper: np.ndarray, budget: int,
                  seed: int | None = None, batch: int = 5000) -> tuple[np.ndarray, float]:
    """Linha de base: amostra ``budget`` soluções aleatórias e devolve a melhor."""
    rng = np.random.default_rng(seed)
    gene_upper = np.asarray(gene_upper, dtype=np.int64)
    best_ind, best_fit = None, np.inf
    done = 0
    while done < budget:
        n = min(batch, budget - done)
        pop = init_population(rng, n, gene_upper)
        fit = fitness_fn(pop)
        i = int(np.argmin(fit))
        if fit[i] < best_fit:
            best_fit, best_ind = float(fit[i]), pop[i].copy()
        done += n
    return best_ind, best_fit

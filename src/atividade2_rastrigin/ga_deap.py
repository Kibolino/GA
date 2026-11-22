"""Algoritmo Genético para a Rastrigin usando DEAP.

Representação
-------------
Cromossomo real: lista de n floats ``[x_1, ..., x_n]`` com x_i em [-5,12; 5,12]
(``creator.Individual`` herda de ``list``). A fitness é o próprio f(x), com
peso -1 (minimização). Como o domínio é uma caixa, os operadores escolhidos
são as versões *limitadas* do DEAP, que nunca geram genes fora do intervalo:

* cruzamento SBX limitado (``tools.cxSimulatedBinaryBounded``, índice eta_c);
* mutação polinomial limitada (``tools.mutPolynomialBounded``, índice eta_m,
  probabilidade ``indpb`` por gene);
* seleção por torneio (``tools.selTournament``).

Elitismo
--------
O ``algorithms.eaSimple`` do DEAP não reinsere os melhores (o HallOfFame só os
*guarda*). Por isso ``ea_elitist`` reescreve o laço geracional: copia os ``e``
melhores (``tools.selBest``) e completa a população com ``N - e`` filhos gerados
por ``algorithms.varAnd``. O restante (estatísticas, Logbook, HallOfFame) é o
padrão da biblioteca.
"""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from deap import algorithms, base, creator, tools

from .rastrigin import LOW, UP, rastrigin_py

# creator.create registra classes globais; o guard evita recriá-las em
# reimportações (p.ex. em testes ou notebooks).
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


@dataclass(frozen=True)
class DeapParams:
    n_dim: int = 2
    pop_size: int = 100
    generations: int = 100
    tournament_size: int = 3
    cxpb: float = 0.9            # prob. de um par de pais cruzar
    mutpb: float = 1.0           # prob. de um filho passar pelo operador de mutação
    indpb: float | None = None   # prob. de mutação por gene (None -> 1/n)
    eta_c: float = 15.0          # índice de distribuição do SBX
    eta_m: float = 20.0          # índice de distribuição da mutação polinomial
    elitism: int = 2

    @property
    def gene_mut_prob(self) -> float:
        return self.indpb if self.indpb is not None else 1.0 / self.n_dim

    @property
    def evaluations_upper_bound(self) -> int:
        """Máximo de avaliações (só indivíduos alterados são reavaliados)."""
        return self.pop_size + self.generations * (self.pop_size - self.elitism)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["indpb"] = self.gene_mut_prob
        return d


@dataclass
class DeapResult:
    best: list[float]
    best_fitness: float
    logbook: tools.Logbook
    history: dict[str, np.ndarray]
    evaluations: int
    elapsed_s: float
    seed: int | None
    snapshots: dict[int, np.ndarray] = field(default_factory=dict)

    def first_hit(self, threshold: float) -> int | None:
        hits = np.flatnonzero(self.history["best"] < threshold)
        return int(hits[0]) if hits.size else None


def evaluate(individual) -> tuple[float]:
    """DEAP espera uma tupla de fitness (mesmo com um único objetivo)."""
    return (rastrigin_py(individual),)


def build_toolbox(p: DeapParams) -> base.Toolbox:
    tb = base.Toolbox()
    # Genes: uniformes no domínio. initRepeat chama attr_float n vezes.
    tb.register("attr_float", random.uniform, LOW, UP)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.attr_float, n=p.n_dim)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", evaluate)
    tb.register("mate", tools.cxSimulatedBinaryBounded, eta=p.eta_c, low=LOW, up=UP)
    tb.register("mutate", tools.mutPolynomialBounded, eta=p.eta_m, low=LOW, up=UP,
                indpb=p.gene_mut_prob)
    tb.register("select", tools.selTournament, tournsize=p.tournament_size)
    return tb


def make_stats() -> tools.Statistics:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("std", np.std)
    return stats


def _evaluate_invalid(toolbox: base.Toolbox, population) -> int:
    invalid = [ind for ind in population if not ind.fitness.valid]
    for ind, fit in zip(invalid, toolbox.map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit
    return len(invalid)


def ea_elitist(population, toolbox: base.Toolbox, cxpb: float, mutpb: float, ngen: int,
               elitism: int, stats: tools.Statistics, halloffame: tools.HallOfFame | None = None,
               snapshot_gens: tuple[int, ...] = ()):
    """Laço geracional com elitismo explícito (variação do ``eaSimple`` do DEAP)."""
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields
    snapshots: dict[int, np.ndarray] = {}

    nevals = _evaluate_invalid(toolbox, population)
    total = nevals
    if halloffame is not None:
        halloffame.update(population)
    logbook.record(gen=0, nevals=nevals, **stats.compile(population))
    if 0 in snapshot_gens:
        snapshots[0] = np.array(population)

    for gen in range(1, ngen + 1):
        elites = [toolbox.clone(ind) for ind in tools.selBest(population, elitism)]
        offspring = toolbox.select(population, len(population) - elitism)
        # varAnd clona os pais e aplica cruzamento (cxpb) e mutação (mutpb),
        # invalidando a fitness apenas dos indivíduos alterados.
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        nevals = _evaluate_invalid(toolbox, offspring)
        total += nevals
        population[:] = elites + offspring
        if halloffame is not None:
            halloffame.update(population)
        logbook.record(gen=gen, nevals=nevals, **stats.compile(population))
        if gen in snapshot_gens:
            snapshots[gen] = np.array(population)
    return population, logbook, total, snapshots


def run_deap(p: DeapParams, seed: int | None = None,
             snapshot_gens: tuple[int, ...] = ()) -> DeapResult:
    """Uma execução completa e reprodutível (DEAP usa o módulo ``random``)."""
    random.seed(seed)
    np.random.seed(None if seed is None else seed % (2**32))
    tb = build_toolbox(p)
    pop = tb.population(n=p.pop_size)
    hof = tools.HallOfFame(1)
    t0 = time.perf_counter()
    pop, log, nevals, snaps = ea_elitist(pop, tb, p.cxpb, p.mutpb, p.generations, p.elitism,
                                          make_stats(), hof, snapshot_gens)
    elapsed = time.perf_counter() - t0
    hist = {
        "best": np.array(log.select("min")),
        "mean": np.array(log.select("avg")),
        "worst": np.array(log.select("max")),
        "std": np.array(log.select("std")),
    }
    return DeapResult(best=list(hof[0]), best_fitness=float(hof[0].fitness.values[0]),
                      logbook=log, history=hist, evaluations=nevals, elapsed_s=elapsed,
                      seed=seed, snapshots=snaps)


def random_search(n_dim: int, budget: int, seed: int | None = None, batch: int = 20000) -> float:
    """Linha de base: melhor valor de f entre ``budget`` pontos uniformes no domínio."""
    from .rastrigin import rastrigin
    rng = np.random.default_rng(seed)
    best, done = np.inf, 0
    while done < budget:
        m = min(batch, budget - done)
        x = rng.uniform(LOW, UP, size=(m, n_dim))
        best = min(best, float(rastrigin(x).min()))
        done += m
    return best

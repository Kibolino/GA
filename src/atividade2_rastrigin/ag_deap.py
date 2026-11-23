from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from deap import algorithms, base, creator, tools

from .rastrigin import LIM_INF, LIM_SUP, rastrigin, rastrigin_individuo

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individuo"):
    creator.create("Individuo", list, fitness=creator.FitnessMin)


@dataclass(frozen=True)
class ParametrosDeap:
    dimensao: int = 2
    tam_populacao: int = 100
    geracoes: int = 100
    tam_torneio: int = 3
    prob_cruzamento: float = 0.9
    prob_mutacao: float = 1.0
    prob_gene: float | None = None
    eta_c: float = 15.0
    eta_m: float = 20.0
    elitismo: int = 2

    @property
    def prob_mutacao_gene(self) -> float:
        return self.prob_gene if self.prob_gene is not None else 1.0 / self.dimensao

    @property
    def limite_avaliacoes(self) -> int:
        return self.tam_populacao + self.geracoes * (self.tam_populacao - self.elitismo)

    def como_dict(self) -> dict:
        d = asdict(self)
        d["prob_gene"] = self.prob_mutacao_gene
        return d


@dataclass
class ResultadoDeap:
    melhor: list[float]
    melhor_fitness: float
    logbook: tools.Logbook
    historico: dict[str, np.ndarray]
    avaliacoes: int
    tempo_s: float
    seed: int | None
    retratos: dict[int, np.ndarray] = field(default_factory=dict)

    def geracao_abaixo_de(self, limiar: float) -> int | None:
        acertos = np.flatnonzero(self.historico["melhor"] < limiar)
        return int(acertos[0]) if acertos.size else None


def avaliar(individuo) -> tuple[float]:
    return (rastrigin_individuo(individuo),)


def criar_toolbox(p: ParametrosDeap) -> base.Toolbox:
    tb = base.Toolbox()
    tb.register("gene", random.uniform, LIM_INF, LIM_SUP)
    tb.register("individuo", tools.initRepeat, creator.Individuo, tb.gene, n=p.dimensao)
    tb.register("populacao", tools.initRepeat, list, tb.individuo)
    tb.register("evaluate", avaliar)
    tb.register("mate", tools.cxSimulatedBinaryBounded, eta=p.eta_c, low=LIM_INF, up=LIM_SUP)
    tb.register("mutate", tools.mutPolynomialBounded, eta=p.eta_m, low=LIM_INF, up=LIM_SUP,
                indpb=p.prob_mutacao_gene)
    tb.register("select", tools.selTournament, tournsize=p.tam_torneio)
    return tb


def criar_estatisticas() -> tools.Statistics:
    estat = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    estat.register("melhor", np.min)
    estat.register("media", np.mean)
    estat.register("pior", np.max)
    estat.register("desvio", np.std)
    return estat


def _avaliar_invalidos(toolbox: base.Toolbox, populacao) -> int:
    invalidos = [ind for ind in populacao if not ind.fitness.valid]
    for ind, fit in zip(invalidos, toolbox.map(toolbox.evaluate, invalidos)):
        ind.fitness.values = fit
    return len(invalidos)


def ag_elitista(populacao, toolbox: base.Toolbox, prob_cruzamento: float, prob_mutacao: float,
                n_geracoes: int, elitismo: int, estatisticas: tools.Statistics,
                hall_da_fama: tools.HallOfFame | None = None,
                geracoes_retrato: tuple[int, ...] = ()):
    logbook = tools.Logbook()
    logbook.header = ["geracao", "avaliacoes"] + estatisticas.fields
    retratos: dict[int, np.ndarray] = {}

    n_aval = _avaliar_invalidos(toolbox, populacao)
    total = n_aval
    if hall_da_fama is not None:
        hall_da_fama.update(populacao)
    logbook.record(geracao=0, avaliacoes=n_aval, **estatisticas.compile(populacao))
    if 0 in geracoes_retrato:
        retratos[0] = np.array(populacao)

    for g in range(1, n_geracoes + 1):
        elite = [toolbox.clone(ind) for ind in tools.selBest(populacao, elitismo)]
        filhos = toolbox.select(populacao, len(populacao) - elitismo)
        filhos = algorithms.varAnd(filhos, toolbox, prob_cruzamento, prob_mutacao)
        n_aval = _avaliar_invalidos(toolbox, filhos)
        total += n_aval
        populacao[:] = elite + filhos
        if hall_da_fama is not None:
            hall_da_fama.update(populacao)
        logbook.record(geracao=g, avaliacoes=n_aval, **estatisticas.compile(populacao))
        if g in geracoes_retrato:
            retratos[g] = np.array(populacao)
    return populacao, logbook, total, retratos


def executar_deap(p: ParametrosDeap, seed: int | None = None,
                  geracoes_retrato: tuple[int, ...] = ()) -> ResultadoDeap:
    random.seed(seed)
    np.random.seed(None if seed is None else seed % (2**32))
    tb = criar_toolbox(p)
    pop = tb.populacao(n=p.tam_populacao)
    hall = tools.HallOfFame(1)
    t0 = time.perf_counter()
    pop, log, n_aval, retratos = ag_elitista(pop, tb, p.prob_cruzamento, p.prob_mutacao, p.geracoes,
                                             p.elitismo, criar_estatisticas(), hall, geracoes_retrato)
    tempo = time.perf_counter() - t0
    hist = {
        "melhor": np.array(log.select("melhor")),
        "medio": np.array(log.select("media")),
        "pior": np.array(log.select("pior")),
        "desvio": np.array(log.select("desvio")),
    }
    return ResultadoDeap(melhor=list(hall[0]), melhor_fitness=float(hall[0].fitness.values[0]),
                         logbook=log, historico=hist, avaliacoes=n_aval, tempo_s=tempo,
                         seed=seed, retratos=retratos)


def busca_aleatoria(dimensao: int, orcamento: int, seed: int | None = None, lote: int = 20000) -> float:
    rng = np.random.default_rng(seed)
    melhor, feitos = np.inf, 0
    while feitos < orcamento:
        m = min(lote, orcamento - feitos)
        x = rng.uniform(LIM_INF, LIM_SUP, size=(m, dimensao))
        melhor = min(melhor, float(rastrigin(x).min()))
        feitos += m
    return melhor

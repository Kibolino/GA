"""Testes da Atividade 1: leitura dos dados, fitness e validade dos operadores."""
import numpy as np
import pytest

from src.atividade1_voos.exact import exact_optimum, pareto_mask
from src.atividade1_voos.flights import hhmm_to_min, load_flights, min_to_hhmm
from src.atividade1_voos.ga import (GAParams, block_shift, crossover, init_population,
                                    mutate, run_ga, tournament_select)
from src.atividade1_voos.model import SCENARIO_A, SCENARIO_B, describe_solution, evaluate


@pytest.fixture(scope="module")
def table():
    return load_flights()


def test_time_conversion():
    assert hhmm_to_min("6:19") == 379
    assert min_to_hhmm(379) == "06:19"


def test_dataset_structure(table):
    assert table.origins == ("LIS", "MAD", "CDG", "DUB", "BRU", "LHR")
    assert table.n_genes == 12
    assert table.gene_upper.tolist() == [10] * 12
    # o exemplo do enunciado deve existir: MAD,FCO,11:01,12:39,260
    mad = [f for f in table.outbound[1] if f.dep == hhmm_to_min("11:01")]
    assert mad and mad[0].arr == hhmm_to_min("12:39") and mad[0].price == 260
    # rotas ordenadas por horário de partida
    for flights in table.outbound + table.inbound:
        deps = [f.dep for f in flights]
        assert deps == sorted(deps)


def test_fitness_hand_computed(table):
    """Confere a fitness do cenário A com um cálculo manual independente."""
    ind = np.zeros(12, dtype=int)  # primeiro voo de cada rota
    arr = [table.outbound[i][0].arr for i in range(6)]
    dep = [table.inbound[i][0].dep for i in range(6)]
    price = sum(table.outbound[i][0].price + table.inbound[i][0].price for i in range(6))
    wait_out = sum(max(arr) - a for a in arr)
    drop = min(dep) - SCENARIO_A.checkin_min
    wait_ret = sum(d - drop for d in dep)
    ev = evaluate(table, SCENARIO_A, ind[None, :])
    assert ev.cost[0] == price
    assert ev.wait_out[0] == wait_out and ev.wait_ret[0] == wait_ret
    assert ev.fitness[0] == pytest.approx(price + SCENARIO_A.wait_weight * (wait_out + wait_ret))


def test_scenario_b_penalizes_infeasible(table):
    ind = np.full(12, 9)  # últimos voos: chegada tarde demais para a mesa-redonda
    ev = evaluate(table, SCENARIO_B, ind[None, :])
    assert ev.penalty[0] > 0 and not ev.feasible[0]


def test_operators_never_create_flights(table):
    """Cruzamento e mutações devem manter todo gene em [0, gene_upper)."""
    rng = np.random.default_rng(0)
    up = table.gene_upper
    pop = init_population(rng, 500, up)
    for kind in ("uniform", "one_point", "two_point"):
        c1, c2 = crossover(rng, pop[:250], pop[250:], kind, 1.0)
        # cada gene do filho vem do mesmo locus de um dos pais
        assert np.all((c1 == pop[:250]) | (c1 == pop[250:]))
        assert np.all((c2 == pop[:250]) | (c2 == pop[250:]))
    m = mutate(rng, pop, up, prob=1.0, creep_prob=0.5)
    assert np.all((m >= 0) & (m < up)) and np.all(m != pop)
    groups = (np.arange(0, 12, 2), np.arange(1, 12, 2))
    s = block_shift(rng, pop, up, 1.0, groups)
    assert np.all((s >= 0) & (s < up))


def test_tournament_prefers_better():
    rng = np.random.default_rng(1)
    fit = np.arange(100, dtype=float)
    win = tournament_select(rng, fit, 10_000, k=3)
    assert fit[win].mean() < fit.mean()


def test_pareto_mask():
    c = np.array([1, 2, 3, 2, 4])
    w = np.array([5, 3, 1, 4, 1])
    assert pareto_mask(c, w).tolist() == [True, True, True, False, False]


def test_ga_reproducible_and_reaches_optimum(table):
    from src.atividade1_voos.model import make_fitness
    fit = make_fitness(table, SCENARIO_B)
    p = GAParams(pop_size=150, generations=120, mutation_prob=0.05, shift_prob=0.3, elitism=5)
    groups = (np.arange(0, 12, 2), np.arange(1, 12, 2))
    r1 = run_ga(fit, table.gene_upper, p, seed=123, shift_groups=groups)
    r2 = run_ga(fit, table.gene_upper, p, seed=123, shift_groups=groups)
    assert r1.best_fitness == r2.best_fitness
    assert np.array_equal(r1.history["best"], r2.history["best"])
    assert np.all(np.diff(r1.history["best"]) <= 0)  # elitismo: melhor nunca piora
    ex = exact_optimum(table, SCENARIO_B)
    assert r1.best_fitness >= ex.fitness - 1e-9
    sol = describe_solution(table, SCENARIO_B, ex.individual)
    assert sol["penalty"] == 0  # o ótimo do cenário B é factível

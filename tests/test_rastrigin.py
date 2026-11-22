"""Testes da Atividade 2: função de Rastrigin e AG com DEAP."""
import numpy as np
import pytest

from src.atividade2_rastrigin.ga_deap import DeapParams, build_toolbox, run_deap
from src.atividade2_rastrigin.rastrigin import LOW, UP, rastrigin, rastrigin_py


def test_rastrigin_global_minimum():
    assert rastrigin(np.zeros(10)) == pytest.approx(0.0)
    assert rastrigin_py([0.0, 0.0]) == pytest.approx(0.0)


def test_rastrigin_versions_agree():
    rng = np.random.default_rng(0)
    x = rng.uniform(LOW, UP, size=(50, 7))
    vec = rastrigin(x)
    py = np.array([rastrigin_py(list(r)) for r in x])
    assert np.allclose(vec, py)
    assert np.all(vec >= 0)


def test_local_minimum_near_integer():
    # mínimo local próximo de (1, 0) vale ~1 (0,995)
    assert 0.9 < rastrigin([0.995, 0.0]) < 1.0


def test_operators_respect_bounds():
    p = DeapParams(n_dim=5, pop_size=50)
    tb = build_toolbox(p)
    pop = tb.population(n=50)
    for _ in range(20):
        for a, b in zip(pop[::2], pop[1::2]):
            tb.mate(a, b)
            tb.mutate(a)
            tb.mutate(b)
    arr = np.array(pop)
    assert np.all(arr >= LOW) and np.all(arr <= UP)


def test_deap_run_reproducible_and_converges():
    p = DeapParams(n_dim=2, pop_size=60, generations=60)
    r1, r2 = run_deap(p, seed=7), run_deap(p, seed=7)
    assert r1.best_fitness == r2.best_fitness
    assert np.all(np.diff(r1.history["best"]) <= 1e-12)  # elitismo
    assert r1.best_fitness < 0.5  # bacia do ótimo global

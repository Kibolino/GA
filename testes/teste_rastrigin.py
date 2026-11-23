import numpy as np
import pytest

from src.atividade2_rastrigin.ag_deap import ParametrosDeap, criar_toolbox, executar_deap
from src.atividade2_rastrigin.rastrigin import LIM_INF, LIM_SUP, rastrigin, rastrigin_individuo


def teste_minimo_global():
    assert rastrigin(np.zeros(10)) == pytest.approx(0.0)
    assert rastrigin_individuo([0.0, 0.0]) == pytest.approx(0.0)


def teste_versoes_concordam():
    rng = np.random.default_rng(0)
    x = rng.uniform(LIM_INF, LIM_SUP, size=(50, 7))
    vetorizada = rastrigin(x)
    individual = np.array([rastrigin_individuo(list(linha)) for linha in x])
    assert np.allclose(vetorizada, individual)
    assert np.all(vetorizada >= 0)


def teste_minimo_local_perto_de_inteiro():
    assert 0.9 < rastrigin([0.995, 0.0]) < 1.0


def teste_operadores_respeitam_limites():
    p = ParametrosDeap(dimensao=5, tam_populacao=50)
    tb = criar_toolbox(p)
    pop = tb.populacao(n=50)
    for _ in range(20):
        for a, b in zip(pop[::2], pop[1::2]):
            tb.mate(a, b)
            tb.mutate(a)
            tb.mutate(b)
    matriz = np.array(pop)
    assert np.all(matriz >= LIM_INF) and np.all(matriz <= LIM_SUP)


def teste_deap_reprodutivel_e_converge():
    p = ParametrosDeap(dimensao=2, tam_populacao=60, geracoes=60)
    r1, r2 = executar_deap(p, seed=7), executar_deap(p, seed=7)
    assert r1.melhor_fitness == r2.melhor_fitness
    assert np.all(np.diff(r1.historico["melhor"]) <= 1e-12)
    assert r1.melhor_fitness < 0.5

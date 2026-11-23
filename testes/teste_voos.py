import numpy as np
import pytest

from src.atividade1_voos.ag import (ParametrosAG, cruzar, deslocar_bloco, executar_ag, mutar,
                                    populacao_inicial, selecao_torneio)
from src.atividade1_voos.exato import mascara_pareto, otimo_exato
from src.atividade1_voos.modelo import (CENARIO_A, CENARIO_B, avaliar, criar_fitness,
                                        descrever_solucao)
from src.atividade1_voos.voos import carregar_voos, hhmm_para_min, min_para_hhmm


@pytest.fixture(scope="module")
def tabela():
    return carregar_voos()


def teste_conversao_de_horario():
    assert hhmm_para_min("6:19") == 379
    assert min_para_hhmm(379) == "06:19"


def teste_estrutura_dos_dados(tabela):
    assert tabela.origens == ("LIS", "MAD", "CDG", "DUB", "BRU", "LHR")
    assert tabela.n_genes == 12
    assert tabela.limite_genes.tolist() == [10] * 12
    mad = [v for v in tabela.ida[1] if v.partida == hhmm_para_min("11:01")]
    assert mad and mad[0].chegada == hhmm_para_min("12:39") and mad[0].preco == 260
    for voos in tabela.ida + tabela.volta:
        partidas = [v.partida for v in voos]
        assert partidas == sorted(partidas)


def teste_fitness_calculada_a_mao(tabela):
    individuo = np.zeros(12, dtype=int)
    chegadas = [tabela.ida[i][0].chegada for i in range(6)]
    partidas = [tabela.volta[i][0].partida for i in range(6)]
    preco = sum(tabela.ida[i][0].preco + tabela.volta[i][0].preco for i in range(6))
    espera_ida = sum(max(chegadas) - a for a in chegadas)
    entrega = min(partidas) - CENARIO_A.checkin_min
    espera_volta = sum(d - entrega for d in partidas)
    aval = avaliar(tabela, CENARIO_A, individuo[None, :])
    assert aval.custo[0] == preco
    assert aval.espera_ida[0] == espera_ida and aval.espera_volta[0] == espera_volta
    assert aval.fitness[0] == pytest.approx(preco + CENARIO_A.peso_espera * (espera_ida + espera_volta))


def teste_cenario_b_penaliza_infactivel(tabela):
    individuo = np.full(12, 9)
    aval = avaliar(tabela, CENARIO_B, individuo[None, :])
    assert aval.penalidade[0] > 0 and not aval.factivel[0]


def teste_operadores_nao_criam_voos(tabela):
    rng = np.random.default_rng(0)
    limite = tabela.limite_genes
    pop = populacao_inicial(rng, 500, limite)
    for tipo in ("uniforme", "um_ponto", "dois_pontos"):
        f1, f2 = cruzar(rng, pop[:250], pop[250:], tipo, 1.0)
        assert np.all((f1 == pop[:250]) | (f1 == pop[250:]))
        assert np.all((f2 == pop[:250]) | (f2 == pop[250:]))
    m = mutar(rng, pop, limite, prob=1.0, prob_creep=0.5)
    assert np.all((m >= 0) & (m < limite)) and np.all(m != pop)
    grupos = (np.arange(0, 12, 2), np.arange(1, 12, 2))
    d = deslocar_bloco(rng, pop, limite, 1.0, grupos)
    assert np.all((d >= 0) & (d < limite))


def teste_torneio_favorece_melhores():
    rng = np.random.default_rng(1)
    fit = np.arange(100, dtype=float)
    vencedores = selecao_torneio(rng, fit, 10_000, k=3)
    assert fit[vencedores].mean() < fit.mean()


def teste_mascara_pareto():
    c = np.array([1, 2, 3, 2, 4])
    w = np.array([5, 3, 1, 4, 1])
    assert mascara_pareto(c, w).tolist() == [True, True, True, False, False]


def teste_ag_reprodutivel_e_atinge_otimo(tabela):
    fitness = criar_fitness(tabela, CENARIO_B)
    p = ParametrosAG(tam_populacao=150, geracoes=120, prob_mutacao=0.05, prob_deslocamento=0.3,
                     elitismo=5)
    grupos = (np.arange(0, 12, 2), np.arange(1, 12, 2))
    r1 = executar_ag(fitness, tabela.limite_genes, p, seed=123, grupos_deslocamento=grupos)
    r2 = executar_ag(fitness, tabela.limite_genes, p, seed=123, grupos_deslocamento=grupos)
    assert r1.melhor_fitness == r2.melhor_fitness
    assert np.array_equal(r1.historico["melhor"], r2.historico["melhor"])
    assert np.all(np.diff(r1.historico["melhor"]) <= 0)
    ex = otimo_exato(tabela, CENARIO_B)
    assert r1.melhor_fitness >= ex.fitness - 1e-9
    sol = descrever_solucao(tabela, CENARIO_B, ex.individuo)
    assert sol["penalidade"] == 0

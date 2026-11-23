from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

FuncaoFitness = Callable[[np.ndarray], np.ndarray]
TIPOS_CRUZAMENTO = ("uniforme", "um_ponto", "dois_pontos")


@dataclass(frozen=True)
class ParametrosAG:
    tam_populacao: int = 100
    geracoes: int = 150
    tam_torneio: int = 3
    cruzamento: str = "uniforme"
    prob_cruzamento: float = 0.9
    prob_mutacao: float = 1 / 12
    prob_creep: float = 0.5
    prob_deslocamento: float = 0.0
    elitismo: int = 2

    def __post_init__(self):
        if self.cruzamento not in TIPOS_CRUZAMENTO:
            raise ValueError(f"cruzamento deve ser um de {TIPOS_CRUZAMENTO}")
        if not 0 <= self.elitismo < self.tam_populacao:
            raise ValueError("elitismo deve estar em [0, tam_populacao)")
        if self.tam_torneio < 1:
            raise ValueError("tam_torneio >= 1")
        for nome in ("prob_cruzamento", "prob_mutacao", "prob_creep", "prob_deslocamento"):
            if not 0.0 <= getattr(self, nome) <= 1.0:
                raise ValueError(f"{nome} deve estar em [0, 1]")

    @property
    def avaliacoes(self) -> int:
        return self.tam_populacao + self.geracoes * (self.tam_populacao - self.elitismo)

    def como_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResultadoAG:
    melhor_individuo: np.ndarray
    melhor_fitness: float
    historico: dict[str, np.ndarray]
    populacao_final: np.ndarray
    fitness_final: np.ndarray
    avaliacoes: int
    tempo_s: float
    seed: int | None = None

    def geracao_que_atingiu(self, alvo: float, tol: float = 1e-9) -> int | None:
        acertos = np.flatnonzero(self.historico["melhor"] <= alvo + tol)
        return int(acertos[0]) if acertos.size else None


def populacao_inicial(rng: np.random.Generator, n: int, limite_genes: np.ndarray) -> np.ndarray:
    return rng.integers(0, limite_genes[None, :], size=(n, limite_genes.size))


def selecao_torneio(rng: np.random.Generator, fitness: np.ndarray, n_selecionados: int,
                    k: int) -> np.ndarray:
    candidatos = rng.integers(0, fitness.size, size=(n_selecionados, k))
    vencedores = np.argmin(fitness[candidatos], axis=1)
    return candidatos[np.arange(n_selecionados), vencedores]


def cruzar(rng: np.random.Generator, pai1: np.ndarray, pai2: np.ndarray, tipo: str,
           prob: float) -> tuple[np.ndarray, np.ndarray]:
    m, L = pai1.shape
    if tipo == "uniforme":
        mascara = rng.random((m, L)) < 0.5
    elif tipo == "um_ponto":
        corte = rng.integers(1, L, size=m)
        mascara = np.arange(L)[None, :] >= corte[:, None]
    elif tipo == "dois_pontos":
        a = rng.integers(1, L, size=m)
        b = rng.integers(1, L, size=m)
        inicio, fim = np.minimum(a, b), np.maximum(a, b)
        posicao = np.arange(L)[None, :]
        mascara = (posicao >= inicio[:, None]) & (posicao < fim[:, None])
    else:
        raise ValueError(tipo)
    mascara &= (rng.random(m) < prob)[:, None]
    filho1 = np.where(mascara, pai2, pai1)
    filho2 = np.where(mascara, pai1, pai2)
    return filho1, filho2


def mutar(rng: np.random.Generator, populacao: np.ndarray, limite_genes: np.ndarray,
          prob: float, prob_creep: float) -> np.ndarray:
    saida = populacao.copy()
    sorteados = rng.random(populacao.shape) < prob
    if not sorteados.any():
        return saida
    limite = np.broadcast_to(limite_genes, populacao.shape)
    creep = rng.random(populacao.shape) < prob_creep

    passo = np.where(rng.random(populacao.shape) < 0.5, -1, 1)
    vizinho = populacao + passo
    vizinho = np.where((vizinho < 0) | (vizinho >= limite), populacao - passo, vizinho)

    r = rng.integers(0, np.maximum(limite - 1, 1))
    r = r + (r >= populacao)
    novo_voo = np.where(limite > 1, r, populacao)

    novo = np.where(creep, vizinho, novo_voo)
    saida[sorteados] = novo[sorteados]
    return saida


def deslocar_bloco(rng: np.random.Generator, populacao: np.ndarray, limite_genes: np.ndarray,
                   prob: float, grupos: tuple[np.ndarray, ...]) -> np.ndarray:
    saida = populacao.copy()
    if prob <= 0 or not grupos:
        return saida
    quem = np.flatnonzero(rng.random(populacao.shape[0]) < prob)
    if quem.size == 0:
        return saida
    grupo_escolhido = rng.integers(0, len(grupos), size=quem.size)
    passos = np.where(rng.random(quem.size) < 0.5, -1, 1)
    for gi, grupo in enumerate(grupos):
        linhas = quem[grupo_escolhido == gi]
        if linhas.size == 0:
            continue
        passo = passos[grupo_escolhido == gi][:, None]
        bloco = saida[np.ix_(linhas, grupo)] + passo
        saida[np.ix_(linhas, grupo)] = np.clip(bloco, 0, limite_genes[grupo] - 1)
    return saida


def executar_ag(funcao_fitness: FuncaoFitness, limite_genes: np.ndarray, params: ParametrosAG,
                seed: int | None = None, grupos_deslocamento: tuple[np.ndarray, ...] = ()
                ) -> ResultadoAG:
    rng = np.random.default_rng(seed)
    limite_genes = np.asarray(limite_genes, dtype=np.int64)
    N, e = params.tam_populacao, params.elitismo
    n_filhos = N - e
    n_pares = (n_filhos + 1) // 2

    t0 = time.perf_counter()
    pop = populacao_inicial(rng, N, limite_genes)
    fit = funcao_fitness(pop)
    avaliacoes = N

    G = params.geracoes
    hist = {k: np.empty(G + 1) for k in ("melhor", "medio", "pior", "desvio", "unicos")}

    def registrar(g: int) -> None:
        hist["melhor"][g] = fit.min()
        hist["medio"][g] = fit.mean()
        hist["pior"][g] = fit.max()
        hist["desvio"][g] = fit.std()
        hist["unicos"][g] = np.unique(pop, axis=0).shape[0]

    melhor_ind = pop[np.argmin(fit)].copy()
    melhor_fit = float(fit.min())
    registrar(0)

    for g in range(1, G + 1):
        ordem = np.argsort(fit, kind="stable")
        elite_pop, elite_fit = pop[ordem[:e]], fit[ordem[:e]]

        pais = selecao_torneio(rng, fit, 2 * n_pares, params.tam_torneio)
        f1, f2 = cruzar(rng, pop[pais[0::2]], pop[pais[1::2]],
                        params.cruzamento, params.prob_cruzamento)
        filhos = np.vstack([f1, f2])[:n_filhos]
        filhos = mutar(rng, filhos, limite_genes, params.prob_mutacao, params.prob_creep)
        filhos = deslocar_bloco(rng, filhos, limite_genes, params.prob_deslocamento,
                                grupos_deslocamento)
        fit_filhos = funcao_fitness(filhos)
        avaliacoes += filhos.shape[0]

        pop = np.vstack([elite_pop, filhos])
        fit = np.concatenate([elite_fit, fit_filhos])

        i = int(np.argmin(fit))
        if fit[i] < melhor_fit:
            melhor_fit, melhor_ind = float(fit[i]), pop[i].copy()
        registrar(g)

    return ResultadoAG(
        melhor_individuo=melhor_ind,
        melhor_fitness=melhor_fit,
        historico=hist,
        populacao_final=pop,
        fitness_final=fit,
        avaliacoes=avaliacoes,
        tempo_s=time.perf_counter() - t0,
        seed=seed,
    )


def busca_aleatoria(funcao_fitness: FuncaoFitness, limite_genes: np.ndarray, orcamento: int,
                    seed: int | None = None, lote: int = 5000) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    limite_genes = np.asarray(limite_genes, dtype=np.int64)
    melhor_ind, melhor_fit = None, np.inf
    feitos = 0
    while feitos < orcamento:
        n = min(lote, orcamento - feitos)
        pop = populacao_inicial(rng, n, limite_genes)
        fit = funcao_fitness(pop)
        i = int(np.argmin(fit))
        if fit[i] < melhor_fit:
            melhor_fit, melhor_ind = float(fit[i]), pop[i].copy()
        feitos += n
    return melhor_ind, melhor_fit

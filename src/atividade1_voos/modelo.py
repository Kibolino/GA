from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .voos import TabelaVoos, min_para_hhmm


@dataclass(frozen=True)
class Cenario:
    chave: str
    nome: str
    peso_espera: float = 1.0
    checkin_min: int = 60
    translado_min: int = 45
    inicio_evento: int | None = None
    fim_evento: int | None = None
    penalidade_fixa: float = 1000.0
    penalidade_por_min: float = 10.0

    @property
    def mesmo_dia(self) -> bool:
        return self.inicio_evento is not None and self.fim_evento is not None

    def com_peso(self, peso_espera: float) -> "Cenario":
        return replace(self, peso_espera=peso_espera)

    def descrever(self) -> str:
        texto = (f"Cenario {self.chave}: {self.nome} | lambda={self.peso_espera} EUR/min, "
                 f"check-in={self.checkin_min} min, translado={self.translado_min} min")
        if self.mesmo_dia:
            texto += (f", mesa-redonda {min_para_hhmm(self.inicio_evento)}-"
                      f"{min_para_hhmm(self.fim_evento)}")
        return texto


CENARIO_A = Cenario(chave="A", nome="ida e volta em dias distintos (sem janela)")
CENARIO_B = Cenario(chave="B", nome="mesa-redonda no mesmo dia (14:00-16:00)",
                    inicio_evento=14 * 60, fim_evento=16 * 60)
CENARIOS = {"A": CENARIO_A, "B": CENARIO_B}


@dataclass
class AvaliacaoSentido:
    custo: np.ndarray
    espera: np.ndarray
    violacao_min: np.ndarray
    n_violacoes: np.ndarray

    def fitness(self, cen: Cenario) -> np.ndarray:
        return (self.custo + cen.peso_espera * self.espera
                + cen.penalidade_fixa * self.n_violacoes + cen.penalidade_por_min * self.violacao_min)


@dataclass
class Avaliacao:
    custo: np.ndarray
    espera_ida: np.ndarray
    espera_volta: np.ndarray
    penalidade: np.ndarray
    fitness: np.ndarray

    @property
    def espera(self) -> np.ndarray:
        return self.espera_ida + self.espera_volta

    @property
    def factivel(self) -> np.ndarray:
        return self.penalidade == 0


def avaliar_ida(tabela: TabelaVoos, cen: Cenario, indices_ida: np.ndarray) -> AvaliacaoSentido:
    linhas = np.arange(tabela.n_pessoas)
    chegada = tabela.ida_chegada[linhas, indices_ida]
    preco = tabela.ida_preco[linhas, indices_ida]
    ultima = chegada.max(axis=1)
    espera = (ultima[:, None] - chegada).sum(axis=1)
    if cen.mesmo_dia:
        atraso = np.maximum(0, ultima + cen.translado_min - cen.inicio_evento)
        violacao_min, n_violacoes = atraso, (atraso > 0).astype(np.int64)
    else:
        violacao_min = n_violacoes = np.zeros(len(indices_ida), dtype=np.int64)
    return AvaliacaoSentido(preco.sum(axis=1), espera, violacao_min, n_violacoes)


def avaliar_volta(tabela: TabelaVoos, cen: Cenario, indices_volta: np.ndarray) -> AvaliacaoSentido:
    linhas = np.arange(tabela.n_pessoas)
    partida = tabela.volta_partida[linhas, indices_volta]
    preco = tabela.volta_preco[linhas, indices_volta]
    if cen.mesmo_dia:
        entrega = cen.fim_evento + cen.translado_min
        falta = np.maximum(0, entrega + cen.checkin_min - partida)
        violacao_min, n_violacoes = falta.sum(axis=1), (falta > 0).sum(axis=1)
        espera = np.maximum(0, partida - entrega).sum(axis=1)
    else:
        entrega = partida.min(axis=1) - cen.checkin_min
        espera = (partida - entrega[:, None]).sum(axis=1)
        violacao_min = n_violacoes = np.zeros(len(indices_volta), dtype=np.int64)
    return AvaliacaoSentido(preco.sum(axis=1), espera, violacao_min, n_violacoes)


def avaliar(tabela: TabelaVoos, cen: Cenario, populacao: np.ndarray) -> Avaliacao:
    populacao = np.atleast_2d(populacao)
    ida = avaliar_ida(tabela, cen, populacao[:, 0::2])
    volta = avaliar_volta(tabela, cen, populacao[:, 1::2])
    penalidade = (cen.penalidade_fixa * (ida.n_violacoes + volta.n_violacoes)
                  + cen.penalidade_por_min * (ida.violacao_min + volta.violacao_min)).astype(float)
    return Avaliacao(
        custo=(ida.custo + volta.custo).astype(float),
        espera_ida=ida.espera.astype(float),
        espera_volta=volta.espera.astype(float),
        penalidade=penalidade,
        fitness=(ida.fitness(cen) + volta.fitness(cen)).astype(float),
    )


def criar_fitness(tabela: TabelaVoos, cen: Cenario):
    def fitness(populacao: np.ndarray) -> np.ndarray:
        return avaliar(tabela, cen, populacao).fitness
    return fitness


def descrever_solucao(tabela: TabelaVoos, cen: Cenario, individuo: np.ndarray) -> dict:
    individuo = np.asarray(individuo, dtype=np.int64)
    aval = avaliar(tabela, cen, individuo[None, :])
    chegadas = [tabela.voo(i, "ida", individuo[2 * i]).chegada for i in range(tabela.n_pessoas)]
    partidas = [tabela.voo(i, "volta", individuo[2 * i + 1]).partida for i in range(tabela.n_pessoas)]
    van_saida = max(chegadas)
    van_entrega = ((cen.fim_evento + cen.translado_min) if cen.mesmo_dia
                   else (min(partidas) - cen.checkin_min))
    linhas = []
    for i, origem in enumerate(tabela.origens):
        vi = tabela.voo(i, "ida", individuo[2 * i])
        vv = tabela.voo(i, "volta", individuo[2 * i + 1])
        linhas.append({
            "origem": origem,
            "ida_indice": int(individuo[2 * i]), "ida_partida": vi.partida,
            "ida_chegada": vi.chegada, "ida_preco": vi.preco,
            "espera_ida": van_saida - vi.chegada,
            "volta_indice": int(individuo[2 * i + 1]), "volta_partida": vv.partida,
            "volta_chegada": vv.chegada, "volta_preco": vv.preco,
            "espera_volta": max(0, vv.partida - van_entrega),
        })
    return {
        "cenario": cen.chave,
        "genes": individuo.tolist(),
        "linhas": linhas,
        "van_saida": van_saida,
        "van_no_evento": van_saida + cen.translado_min,
        "van_entrega": van_entrega,
        "custo": float(aval.custo[0]),
        "espera_ida": float(aval.espera_ida[0]),
        "espera_volta": float(aval.espera_volta[0]),
        "espera": float(aval.espera[0]),
        "penalidade": float(aval.penalidade[0]),
        "fitness": float(aval.fitness[0]),
    }


def formatar_solucao(sol: dict) -> str:
    texto = [
        f"Cenario {sol['cenario']} | genes = {sol['genes']}",
        f"{'Origem':<6} {'Ida (partida-chegada)':<22} {'EUR':>4} {'Esp.':>5} | "
        f"{'Volta (partida-chegada)':<24} {'EUR':>4} {'Esp.':>5}",
    ]
    for r in sol["linhas"]:
        texto.append(
            f"{r['origem']:<6} {min_para_hhmm(r['ida_partida']) + ' - ' + min_para_hhmm(r['ida_chegada']):<22} "
            f"{r['ida_preco']:>4} {r['espera_ida']:>5} | "
            f"{min_para_hhmm(r['volta_partida']) + ' - ' + min_para_hhmm(r['volta_chegada']):<24} "
            f"{r['volta_preco']:>4} {r['espera_volta']:>5}")
    texto.append(
        f"Van sai de FCO as {min_para_hhmm(sol['van_saida'])} | van deixa o grupo em FCO as "
        f"{min_para_hhmm(sol['van_entrega'])}")
    texto.append(
        f"Custo = EUR {sol['custo']:.0f} | espera = {sol['espera']:.0f} min "
        f"(ida {sol['espera_ida']:.0f} + volta {sol['espera_volta']:.0f}) | "
        f"penalidade = {sol['penalidade']:.0f} | F = {sol['fitness']:.1f}")
    return "\n".join(texto)

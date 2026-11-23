from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ORIGENS: tuple[str, ...] = ("LIS", "MAD", "CDG", "DUB", "BRU", "LHR")
DESTINO = "FCO"
NOMES_CIDADES = {
    "LIS": "Lisboa", "MAD": "Madri", "CDG": "Paris", "DUB": "Dublin",
    "BRU": "Bruxelas", "LHR": "Londres", "FCO": "Roma",
}

ARQUIVO_PADRAO = Path(__file__).resolve().parents[2] / "data" / "flights.txt"


def hhmm_para_min(texto: str) -> int:
    h, m = texto.strip().split(":")
    return int(h) * 60 + int(m)


def min_para_hhmm(minutos: float) -> str:
    minutos = int(round(minutos))
    return f"{minutos // 60:02d}:{minutos % 60:02d}"


@dataclass(frozen=True)
class Voo:
    origem: str
    destino: str
    partida: int
    chegada: int
    preco: int

    def rotulo(self) -> str:
        return (f"{self.origem}->{self.destino} {min_para_hhmm(self.partida)}-"
                f"{min_para_hhmm(self.chegada)} EUR {self.preco}")


@dataclass(frozen=True)
class TabelaVoos:
    origens: tuple[str, ...]
    ida: tuple[tuple[Voo, ...], ...]
    volta: tuple[tuple[Voo, ...], ...]
    ida_partida: np.ndarray
    ida_chegada: np.ndarray
    ida_preco: np.ndarray
    volta_partida: np.ndarray
    volta_chegada: np.ndarray
    volta_preco: np.ndarray

    @property
    def n_pessoas(self) -> int:
        return len(self.origens)

    @property
    def n_genes(self) -> int:
        return 2 * self.n_pessoas

    @property
    def limite_genes(self) -> np.ndarray:
        limite = np.empty(self.n_genes, dtype=np.int64)
        limite[0::2] = [len(v) for v in self.ida]
        limite[1::2] = [len(v) for v in self.volta]
        return limite

    @property
    def tamanho_espaco(self) -> int:
        return int(np.prod(self.limite_genes.astype(object)))

    def voo(self, pessoa: int, sentido: str, indice: int) -> Voo:
        voos = self.ida[pessoa] if sentido == "ida" else self.volta[pessoa]
        return voos[indice]

    def nomes_genes(self) -> list[str]:
        nomes = []
        for o in self.origens:
            nomes += [f"{o}_ida", f"{o}_volta"]
        return nomes


def _matriz(grupos: list[list[Voo]], atributo: str) -> np.ndarray:
    largura = max(len(g) for g in grupos)
    mat = np.zeros((len(grupos), largura), dtype=np.int64)
    for i, g in enumerate(grupos):
        valores = [getattr(v, atributo) for v in g]
        mat[i, :len(valores)] = valores
        mat[i, len(valores):] = valores[-1]
    return mat


def carregar_voos(caminho: Path | str = ARQUIVO_PADRAO, origens: tuple[str, ...] = ORIGENS,
                  destino: str = DESTINO) -> TabelaVoos:
    caminho = Path(caminho)
    ida: dict[str, list[Voo]] = {o: [] for o in origens}
    volta: dict[str, list[Voo]] = {o: [] for o in origens}
    for num_linha, bruta in enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1):
        linha = bruta.strip()
        if not linha or linha.startswith("#"):
            continue
        partes = [p.strip() for p in linha.split(",")]
        if len(partes) != 5:
            raise ValueError(f"{caminho}:{num_linha}: esperado 5 campos, obtido {len(partes)}")
        org, dst, partida, chegada, preco = partes
        v = Voo(org, dst, hhmm_para_min(partida), hhmm_para_min(chegada), int(preco))
        if dst == destino and org in ida:
            ida[org].append(v)
        elif org == destino and dst in volta:
            volta[dst].append(v)

    for o in origens:
        if not ida[o] or not volta[o]:
            raise ValueError(f"Rota sem voos para {o}: ida={len(ida[o])}, volta={len(volta[o])}")
        ida[o].sort(key=lambda v: v.partida)
        volta[o].sort(key=lambda v: v.partida)
        for v in ida[o] + volta[o]:
            if v.chegada <= v.partida:
                raise ValueError(f"Voo atravessa a meia-noite (não suportado): {v}")

    grupos_ida = [ida[o] for o in origens]
    grupos_volta = [volta[o] for o in origens]
    return TabelaVoos(
        origens=tuple(origens),
        ida=tuple(tuple(g) for g in grupos_ida),
        volta=tuple(tuple(g) for g in grupos_volta),
        ida_partida=_matriz(grupos_ida, "partida"),
        ida_chegada=_matriz(grupos_ida, "chegada"),
        ida_preco=_matriz(grupos_ida, "preco"),
        volta_partida=_matriz(grupos_volta, "partida"),
        volta_chegada=_matriz(grupos_volta, "chegada"),
        volta_preco=_matriz(grupos_volta, "preco"),
    )

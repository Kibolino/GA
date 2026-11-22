"""Leitura do ``flights.txt`` e organização do espaço de busca.

Cada linha do arquivo tem o formato ``ORIGEM,DESTINO,PARTIDA,CHEGADA,PRECO``,
por exemplo ``MAD,FCO,11:01,12:39,260``. Os horários são convertidos para
minutos desde 00:00 (inteiros), o que simplifica as diferenças de tempo.

Para cada pesquisador i (cidade de origem) existem duas listas de voos:

* ``outbound[i]``: voos ORIGEM -> FCO (ida), ordenados por horário de partida;
* ``inbound[i]`` : voos FCO -> ORIGEM (volta), ordenados por horário de partida.

A ordenação por horário é proposital: índices vizinhos correspondem a voos
vizinhos no tempo, o que dá sentido à mutação do tipo "creep" (+/-1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ordem fixa dos pesquisadores (= ordem dos genes no cromossomo).
ORIGINS: tuple[str, ...] = ("LIS", "MAD", "CDG", "DUB", "BRU", "LHR")
DESTINATION = "FCO"
CITY_NAMES = {
    "LIS": "Lisboa", "MAD": "Madri", "CDG": "Paris", "DUB": "Dublin",
    "BRU": "Bruxelas", "LHR": "Londres", "FCO": "Roma",
}

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "flights.txt"


def hhmm_to_min(text: str) -> int:
    """'6:19' -> 379 (minutos desde a meia-noite)."""
    h, m = text.strip().split(":")
    return int(h) * 60 + int(m)


def min_to_hhmm(minutes: float) -> str:
    """379 -> '06:19'. Aceita valores fora de [0, 1440) (útil para exibir esperas)."""
    minutes = int(round(minutes))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


@dataclass(frozen=True)
class Flight:
    origin: str
    dest: str
    dep: int    # partida (min desde 00:00)
    arr: int    # chegada (min desde 00:00)
    price: int  # euros

    def label(self) -> str:
        return (f"{self.origin}->{self.dest} {min_to_hhmm(self.dep)}-"
                f"{min_to_hhmm(self.arr)} EUR {self.price}")


@dataclass(frozen=True)
class FlightTable:
    """Espaço de busca organizado por pesquisador e sentido.

    Além das listas de objetos ``Flight`` (para relatórios), guarda matrizes
    numpy ``(n_pessoas, max_voos)`` usadas na avaliação vetorizada da
    população. Rotas com menos voos que ``max_voos`` são completadas com
    valores de preenchimento, que nunca são acessados porque os operadores
    respeitam ``gene_upper`` (limite superior exclusivo de cada gene).
    """
    origins: tuple[str, ...]
    outbound: tuple[tuple[Flight, ...], ...]
    inbound: tuple[tuple[Flight, ...], ...]
    out_dep: np.ndarray
    out_arr: np.ndarray
    out_price: np.ndarray
    ret_dep: np.ndarray
    ret_arr: np.ndarray
    ret_price: np.ndarray

    @property
    def n_people(self) -> int:
        return len(self.origins)

    @property
    def n_genes(self) -> int:
        return 2 * self.n_people

    @property
    def gene_upper(self) -> np.ndarray:
        """Número de voos disponíveis para cada gene (ordem: ida0, volta0, ida1, ...)."""
        up = np.empty(self.n_genes, dtype=np.int64)
        up[0::2] = [len(f) for f in self.outbound]
        up[1::2] = [len(f) for f in self.inbound]
        return up

    @property
    def search_space_size(self) -> int:
        return int(np.prod(self.gene_upper.astype(object)))

    def flight(self, person: int, leg: str, idx: int) -> Flight:
        """Voo efetivo associado a um gene (``leg`` em {'out', 'ret'})."""
        flights = self.outbound[person] if leg == "out" else self.inbound[person]
        return flights[idx]

    def gene_names(self) -> list[str]:
        names = []
        for o in self.origins:
            names += [f"{o}_ida", f"{o}_volta"]
        return names


def _as_matrix(groups: list[list[Flight]], attr: str) -> np.ndarray:
    width = max(len(g) for g in groups)
    mat = np.zeros((len(groups), width), dtype=np.int64)
    for i, g in enumerate(groups):
        vals = [getattr(f, attr) for f in g]
        mat[i, :len(vals)] = vals
        mat[i, len(vals):] = vals[-1]  # preenchimento (nunca indexado)
    return mat


def load_flights(path: Path | str = DEFAULT_DATA, origins: tuple[str, ...] = ORIGINS,
                 destination: str = DESTINATION) -> FlightTable:
    """Lê o arquivo de voos e monta a ``FlightTable``."""
    path = Path(path)
    out: dict[str, list[Flight]] = {o: [] for o in origins}
    ret: dict[str, list[Flight]] = {o: [] for o in origins}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"{path}:{lineno}: esperado 5 campos, obtido {len(parts)}")
        org, dst, dep, arr, price = parts
        f = Flight(org, dst, hhmm_to_min(dep), hhmm_to_min(arr), int(price))
        if dst == destination and org in out:
            out[org].append(f)
        elif org == destination and dst in ret:
            ret[dst].append(f)
        # Voos que não envolvem os pesquisadores considerados são ignorados.

    for o in origins:
        if not out[o] or not ret[o]:
            raise ValueError(f"Rota sem voos para {o}: ida={len(out[o])}, volta={len(ret[o])}")
        out[o].sort(key=lambda f: f.dep)
        ret[o].sort(key=lambda f: f.dep)
        # Premissa: nenhum voo atravessa a meia-noite (verificado aqui).
        for f in out[o] + ret[o]:
            if f.arr <= f.dep:
                raise ValueError(f"Voo atravessa a meia-noite (não suportado): {f}")

    out_groups = [out[o] for o in origins]
    ret_groups = [ret[o] for o in origins]
    return FlightTable(
        origins=tuple(origins),
        outbound=tuple(tuple(g) for g in out_groups),
        inbound=tuple(tuple(g) for g in ret_groups),
        out_dep=_as_matrix(out_groups, "dep"),
        out_arr=_as_matrix(out_groups, "arr"),
        out_price=_as_matrix(out_groups, "price"),
        ret_dep=_as_matrix(ret_groups, "dep"),
        ret_arr=_as_matrix(ret_groups, "arr"),
        ret_price=_as_matrix(ret_groups, "price"),
    )

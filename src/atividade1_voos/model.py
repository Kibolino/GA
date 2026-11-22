"""Modelagem do problema: cenários (premissas), decodificação e função fitness.

Representação do indivíduo
--------------------------
Vetor de 12 inteiros, dois genes por pesquisador, na ordem de ``ORIGINS``::

    [ida_LIS, volta_LIS, ida_MAD, volta_MAD, ..., ida_LHR, volta_LHR]

O gene ``2i`` é o índice do voo de ida (origem_i -> FCO) e o gene ``2i+1`` o
índice do voo de volta (FCO -> origem_i), ambos em ``[0, n_voos_da_rota)``.
Como todo valor nesse intervalo aponta para uma linha real do
``flights.txt``, qualquer cromossomo é uma solução existente.

Função fitness (minimização)
----------------------------
    F(x) = C(x) + lambda * W(x) + P(x)

* ``C``: soma dos preços das 12 passagens (EUR);
* ``W``: soma, sobre os 6 pesquisadores, do tempo de espera no aeroporto de
  FCO na chegada e na volta (min);
* ``lambda``: custo do tempo de espera (EUR/min), converte minutos em euros;
* ``P``: penalidade das restrições de horário (só no cenário com janela do
  congresso).

Premissas de tempo (ver README e artigo)
----------------------------------------
P1. Uma única van leva o grupo de FCO ao congresso e só sai quando o último
    pesquisador chega: espera de ida = ``ultima_chegada - chegada_i``.
P2. A van deixa o grupo junto em FCO na volta.
    * Cenário A (dias distintos): a van chega ``checkin_min`` antes do
      primeiro voo de volta -> espera = ``partida_i - (min(partidas) - checkin)``.
    * Cenário B (mesa-redonda no mesmo dia): a van sai do congresso no fim da
      mesa (``event_end``) e chega a FCO após ``transfer_min`` ->
      espera = ``partida_i - (event_end + transfer_min)``.
P3. (B) Restrição de chegada: ``ultima_chegada + transfer_min <= event_start``.
P4. (B) Restrição de volta: ``partida_i >= event_end + transfer_min + checkin_min``.
P5. Restrições violadas não descartam o indivíduo: recebem penalidade fixa por
    restrição + penalidade proporcional aos minutos de violação, o que dá ao
    AG um "gradiente" de volta para a região factível.

A fitness é *separável*: F = F_ida(genes de ida) + F_volta(genes de volta).
Isso é explorado em ``exact.py`` para calcular o ótimo global por enumeração.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .flights import FlightTable, min_to_hhmm


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    wait_weight: float = 1.0            # lambda (EUR por minuto de espera)
    checkin_min: int = 60               # antecedência mínima no aeroporto na volta
    transfer_min: int = 45              # trajeto de van FCO <-> local do congresso
    event_start: int | None = None      # início da mesa-redonda (min desde 00:00)
    event_end: int | None = None        # fim da mesa-redonda
    penalty_fixed: float = 1000.0       # EUR por restrição violada
    penalty_per_min: float = 10.0       # EUR por minuto de violação

    @property
    def same_day(self) -> bool:
        return self.event_start is not None and self.event_end is not None

    def with_weight(self, wait_weight: float) -> "Scenario":
        return replace(self, wait_weight=wait_weight)

    def describe(self) -> str:
        base = (f"Cenario {self.key}: {self.name} | lambda={self.wait_weight} EUR/min, "
                f"check-in={self.checkin_min} min, translado={self.transfer_min} min")
        if self.same_day:
            base += (f", mesa-redonda {min_to_hhmm(self.event_start)}-"
                     f"{min_to_hhmm(self.event_end)}")
        return base


SCENARIO_A = Scenario(key="A", name="ida e volta em dias distintos (sem janela)")
SCENARIO_B = Scenario(key="B", name="mesa-redonda no mesmo dia (14:00-16:00)",
                      event_start=14 * 60, event_end=16 * 60)
SCENARIOS = {"A": SCENARIO_A, "B": SCENARIO_B}


@dataclass
class SideEval:
    """Componentes da fitness de um dos sentidos (ida ou volta)."""
    cost: np.ndarray       # EUR
    wait: np.ndarray       # minutos (soma sobre as pessoas)
    viol_min: np.ndarray   # minutos de violação das restrições
    n_viol: np.ndarray     # número de restrições violadas

    def fitness(self, sc: Scenario) -> np.ndarray:
        return (self.cost + sc.wait_weight * self.wait
                + sc.penalty_fixed * self.n_viol + sc.penalty_per_min * self.viol_min)


@dataclass
class Evaluation:
    cost: np.ndarray
    wait_out: np.ndarray
    wait_ret: np.ndarray
    penalty: np.ndarray
    fitness: np.ndarray

    @property
    def wait(self) -> np.ndarray:
        return self.wait_out + self.wait_ret

    @property
    def feasible(self) -> np.ndarray:
        return self.penalty == 0


def evaluate_outbound(table: FlightTable, sc: Scenario, idx_out: np.ndarray) -> SideEval:
    """Avalia os genes de ida. ``idx_out`` tem forma (N, n_pessoas)."""
    rows = np.arange(table.n_people)
    arr = table.out_arr[rows, idx_out]
    price = table.out_price[rows, idx_out]
    last = arr.max(axis=1)                         # horário de saída da van (P1)
    wait = (last[:, None] - arr).sum(axis=1)
    if sc.same_day:
        late = np.maximum(0, last + sc.transfer_min - sc.event_start)   # P3
        viol_min, n_viol = late, (late > 0).astype(np.int64)
    else:
        viol_min = n_viol = np.zeros(len(idx_out), dtype=np.int64)
    return SideEval(price.sum(axis=1), wait, viol_min, n_viol)


def evaluate_return(table: FlightTable, sc: Scenario, idx_ret: np.ndarray) -> SideEval:
    """Avalia os genes de volta. ``idx_ret`` tem forma (N, n_pessoas)."""
    rows = np.arange(table.n_people)
    dep = table.ret_dep[rows, idx_ret]
    price = table.ret_price[rows, idx_ret]
    if sc.same_day:
        drop = sc.event_end + sc.transfer_min                            # P2 (B)
        short = np.maximum(0, drop + sc.checkin_min - dep)               # P4
        viol_min, n_viol = short.sum(axis=1), (short > 0).sum(axis=1)
        wait = np.maximum(0, dep - drop).sum(axis=1)
    else:
        drop = dep.min(axis=1) - sc.checkin_min                          # P2 (A)
        wait = (dep - drop[:, None]).sum(axis=1)
        viol_min = n_viol = np.zeros(len(idx_ret), dtype=np.int64)
    return SideEval(price.sum(axis=1), wait, viol_min, n_viol)


def evaluate(table: FlightTable, sc: Scenario, pop: np.ndarray) -> Evaluation:
    """Avalia uma população inteira (matriz N x 12) de forma vetorizada."""
    pop = np.atleast_2d(pop)
    o = evaluate_outbound(table, sc, pop[:, 0::2])
    r = evaluate_return(table, sc, pop[:, 1::2])
    penalty = (sc.penalty_fixed * (o.n_viol + r.n_viol)
               + sc.penalty_per_min * (o.viol_min + r.viol_min)).astype(float)
    return Evaluation(
        cost=(o.cost + r.cost).astype(float),
        wait_out=o.wait.astype(float),
        wait_ret=r.wait.astype(float),
        penalty=penalty,
        fitness=(o.fitness(sc) + r.fitness(sc)).astype(float),
    )


def make_fitness(table: FlightTable, sc: Scenario):
    """Fecha a tabela e o cenário numa função ``pop -> fitness`` usada pelo AG."""
    def fitness(pop: np.ndarray) -> np.ndarray:
        return evaluate(table, sc, pop).fitness
    return fitness


def describe_solution(table: FlightTable, sc: Scenario, ind: np.ndarray) -> dict:
    """Decodifica um cromossomo em um roteiro legível (usado em relatórios/gráficos)."""
    ind = np.asarray(ind, dtype=np.int64)
    ev = evaluate(table, sc, ind[None, :])
    rows = []
    arrs = [table.flight(i, "out", ind[2 * i]).arr for i in range(table.n_people)]
    deps = [table.flight(i, "ret", ind[2 * i + 1]).dep for i in range(table.n_people)]
    van_out = max(arrs)
    van_drop = (sc.event_end + sc.transfer_min) if sc.same_day else (min(deps) - sc.checkin_min)
    for i, origin in enumerate(table.origins):
        fo = table.flight(i, "out", ind[2 * i])
        fr = table.flight(i, "ret", ind[2 * i + 1])
        rows.append({
            "origin": origin,
            "out_idx": int(ind[2 * i]), "out_dep": fo.dep, "out_arr": fo.arr, "out_price": fo.price,
            "wait_out": van_out - fo.arr,
            "ret_idx": int(ind[2 * i + 1]), "ret_dep": fr.dep, "ret_arr": fr.arr, "ret_price": fr.price,
            "wait_ret": max(0, fr.dep - van_drop),
        })
    return {
        "scenario": sc.key,
        "genes": ind.tolist(),
        "rows": rows,
        "van_out": van_out,
        "van_at_event": van_out + sc.transfer_min,
        "van_drop": van_drop,
        "cost": float(ev.cost[0]),
        "wait_out": float(ev.wait_out[0]),
        "wait_ret": float(ev.wait_ret[0]),
        "wait": float(ev.wait[0]),
        "penalty": float(ev.penalty[0]),
        "fitness": float(ev.fitness[0]),
    }


def format_solution(sol: dict) -> str:
    """Tabela em texto puro de uma solução decodificada."""
    lines = [
        f"Cenario {sol['scenario']} | genes = {sol['genes']}",
        f"{'Origem':<6} {'Ida (partida-chegada)':<22} {'EUR':>4} {'Esp.':>5} | "
        f"{'Volta (partida-chegada)':<24} {'EUR':>4} {'Esp.':>5}",
    ]
    for r in sol["rows"]:
        lines.append(
            f"{r['origin']:<6} {min_to_hhmm(r['out_dep']) + ' - ' + min_to_hhmm(r['out_arr']):<22} "
            f"{r['out_price']:>4} {r['wait_out']:>5} | "
            f"{min_to_hhmm(r['ret_dep']) + ' - ' + min_to_hhmm(r['ret_arr']):<24} "
            f"{r['ret_price']:>4} {r['wait_ret']:>5}")
    lines.append(
        f"Van sai de FCO as {min_to_hhmm(sol['van_out'])} | van deixa o grupo em FCO as "
        f"{min_to_hhmm(sol['van_drop'])}")
    lines.append(
        f"Custo = EUR {sol['cost']:.0f} | espera = {sol['wait']:.0f} min "
        f"(ida {sol['wait_out']:.0f} + volta {sol['wait_ret']:.0f}) | "
        f"penalidade = {sol['penalty']:.0f} | F = {sol['fitness']:.1f}")
    return "\n".join(lines)

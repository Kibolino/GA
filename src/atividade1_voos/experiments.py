"""Experimentos da Atividade 1 (voos) - gera métricas, tabelas LaTeX e gráficos.

Uso (a partir da raiz do projeto)::

    python -m src.atividade1_voos.experiments              # experimento completo
    python -m src.atividade1_voos.experiments --quick      # versão reduzida (teste)
    python -m src.atividade1_voos.experiments --seed 7 --runs 50

Etapas:
1. ótimo exato de cada cenário por enumeração decomposta (validação);
2. 30 execuções independentes do AG por cenário (seeds ``seed .. seed+29``);
3. linha de base: busca aleatória com o mesmo nº de avaliações;
4. análise de sensibilidade (um fator por vez) no cenário A;
5. varredura de lambda (custo x espera) e fronteira de Pareto exata;
6. figuras (PDF/PNG), tabelas (.tex), CSV e JSON em ``results/atividade1``.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..common import plotstyle
from ..common.stats import fmt_num, save_csv, save_json, summarize, write_latex_table
from . import plots
from .exact import enumerate_sides, exact_optimum, pareto_front
from .flights import FlightTable, load_flights, min_to_hhmm
from .ga import GAParams, GAResult, random_search, run_ga
from .model import SCENARIOS, Scenario, describe_solution, format_solution, make_fitness

ROOT = Path(__file__).resolve().parents[2]

# Configuração base (escolhida após um ajuste preliminar; ver artigo, Seção de parâmetros).
BASE_PARAMS = GAParams(
    pop_size=150,
    generations=400,
    tournament_size=3,
    crossover="uniform",
    crossover_prob=0.9,
    mutation_prob=0.05,
    creep_prob=0.5,
    shift_prob=0.3,
    elitism=5,
)

# Fatores da análise de sensibilidade (um fator por vez em torno de BASE_PARAMS).
SENSITIVITY = [
    ("Torneio $k$", "tournament_size", [2, 3, 5]),
    ("Cruzamento", "crossover", ["one_point", "two_point", "uniform"]),
    ("$p_c$", "crossover_prob", [0.6, 0.9, 1.0]),
    ("$p_m$ (gene)", "mutation_prob", [0.02, 0.05, 0.10]),
    ("$p_{\\mathrm{shift}}$", "shift_prob", [0.0, 0.1, 0.3, 0.5]),
    ("Elitismo $e$", "elitism", [0, 1, 5, 10]),
    ("População $N$", "pop_size", [50, 150, 300]),
]
CX_NAMES = {"one_point": "1 ponto", "two_point": "2 pontos", "uniform": "uniforme"}
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def shift_groups(table: FlightTable) -> tuple[np.ndarray, np.ndarray]:
    """Grupos da mutação de bloco: todos os genes de ida e todos os de volta."""
    return np.arange(0, table.n_genes, 2), np.arange(1, table.n_genes, 2)


def run_batch(table: FlightTable, sc: Scenario, params: GAParams, runs: int,
              seed: int) -> list[GAResult]:
    fit = make_fitness(table, sc)
    groups = shift_groups(table)
    return [run_ga(fit, table.gene_upper, params, seed=seed + r, shift_groups=groups)
            for r in range(runs)]


def batch_metrics(results: list[GAResult], f_star: float) -> dict:
    best = [r.best_fitness for r in results]
    hits = [r.first_hit(f_star) for r in results]
    hit_gens = [h for h in hits if h is not None]
    return {
        "best": summarize(best),
        "gap_pct_mean": 100.0 * (np.mean(best) - f_star) / f_star,
        "success": len(hit_gens),
        "runs": len(results),
        "hit_gen": summarize(hit_gens) if hit_gens else None,
        "final_pop_mean": summarize([r.final_fitness.mean() for r in results]),
        "final_pop_std": summarize([r.final_fitness.std(ddof=1) for r in results]),
        "time_s": summarize([r.elapsed_s for r in results]),
    }


# --------------------------------------------------------------------------
# Tabelas LaTeX
# --------------------------------------------------------------------------

def table_params(path: Path, p: GAParams, runs: int, seed: int) -> None:
    sa, sb = SCENARIOS["A"], SCENARIOS["B"]
    rows = [
        ["Representação", "12 inteiros (ida/volta de cada pesquisador), gene $\\in\\{0..9\\}$"],
        ["Tamanho da população $N$", f"{p.pop_size}"],
        ["Número de gerações $G$", f"{p.generations}"],
        ["Seleção", f"torneio determinístico, $k={p.tournament_size}$, com reposição"],
        ["Cruzamento", f"{CX_NAMES[p.crossover]}, $p_c={fmt_num(p.crossover_prob, 2)}$"],
        ["Mutação por gene", f"$p_m={fmt_num(p.mutation_prob, 2)}$ "
                             f"({int(p.creep_prob * 100)}\\% creep $\\pm1$, resto reset)"],
        ["Mutação de bloco", f"$p_{{\\mathrm{{shift}}}}={fmt_num(p.shift_prob, 2)}$ por filho"],
        ["Elitismo", f"{p.elitism} melhores copiados sem alteração"],
        ["Avaliações/execução", fmt_num(p.evaluations, 0)],
        ["Execuções", f"{runs} (seeds {seed}--{seed + runs - 1})"],
        "midrule",
        ["$\\lambda$ (custo da espera)", f"{fmt_num(sa.wait_weight, 0)} EUR/min"],
        ["Check-in / translado", f"{sa.checkin_min} min / {sa.transfer_min} min"],
        ["Janela (cenário B)", f"{min_to_hhmm(sb.event_start)}--{min_to_hhmm(sb.event_end)}"],
        ["Penalidade (cenário B)", f"{fmt_num(sb.penalty_fixed, 0)} EUR/restrição + "
                                   f"{fmt_num(sb.penalty_per_min, 0)} EUR/min"],
    ]
    write_latex_table(path, ["Parâmetro", "Valor"], rows, "@{}lp{4.6cm}@{}",
                      "Parâmetros do AG da Atividade 1 (implementação própria).",
                      "tab:a1_params")


def table_stats(path: Path, summary: dict) -> None:
    keys = list(summary["scenarios"])
    S = summary["scenarios"]

    def row(label, fn):
        return [label] + [fn(S[k]) for k in keys]

    rows = [
        row("Ótimo exato $F^*$ (enumeração)", lambda s: fmt_num(s["exact"]["fitness"], 0)),
        "midrule",
        row("Média do melhor $F$", lambda s: fmt_num(s["ga"]["best"]["mean"], 1)),
        row("Variância", lambda s: fmt_num(s["ga"]["best"]["var"], 1)),
        row("Desvio padrão", lambda s: fmt_num(s["ga"]["best"]["std"], 1)),
        row("Mínimo", lambda s: fmt_num(s["ga"]["best"]["min"], 0)),
        row("Mediana", lambda s: fmt_num(s["ga"]["best"]["median"], 1)),
        row("Máximo", lambda s: fmt_num(s["ga"]["best"]["max"], 0)),
        row("Gap médio para $F^*$", lambda s: fmt_num(s["ga"]["gap_pct_mean"], 2) + "\\%"),
        row("Execuções com $F=F^*$", lambda s: f"{s['ga']['success']}/{s['ga']['runs']}"),
        row("Geração p/ atingir $F^*$ (média)", lambda s: fmt_num(s["ga"]["hit_gen"]["mean"], 1)
            if s["ga"]["hit_gen"] else "--"),
        row("$\\bar F$ da pop. final (média)", lambda s: fmt_num(s["ga"]["final_pop_mean"]["mean"], 1)),
        row("$\\sigma_F$ da pop. final (média)", lambda s: fmt_num(s["ga"]["final_pop_std"]["mean"], 1)),
        row("Tempo por execução (s)", lambda s: fmt_num(s["ga"]["time_s"]["mean"], 2)),
        "midrule",
        row("Busca aleatória: média", lambda s: fmt_num(s["random"]["mean"], 1)),
        row("Busca aleatória: desvio padrão", lambda s: fmt_num(s["random"]["std"], 1)),
    ]
    write_latex_table(path, ["Métrica (30 execuções)"] + [f"Cen. {k}" for k in keys], rows,
                      "@{}l" + "r" * len(keys) + "@{}",
                      "Estatísticas do melhor fitness final $F$ (EUR) do AG na Atividade 1.",
                      "tab:a1_stats",
                      notes="Variância e desvio padrão amostrais ($n-1$). A busca aleatória usa "
                            "o mesmo número de avaliações do AG.")


def table_schedule(path: Path, sol: dict, sc: Scenario) -> None:
    rows = []
    for r in sol["rows"]:
        rows.append([
            r["origin"],
            f"{min_to_hhmm(r['out_dep'])}--{min_to_hhmm(r['out_arr'])}", f"{r['out_price']}",
            f"{r['wait_out']}",
            f"{min_to_hhmm(r['ret_dep'])}--{min_to_hhmm(r['ret_arr'])}", f"{r['ret_price']}",
            f"{r['wait_ret']}",
        ])
    rows.append("midrule")
    rows.append(["Total", "", fmt_num(sum(r["out_price"] for r in sol["rows"]), 0),
                 fmt_num(sol["wait_out"], 0), "",
                 fmt_num(sum(r["ret_price"] for r in sol["rows"]), 0), fmt_num(sol["wait_ret"], 0)])
    van = (f"Van sai de FCO às {min_to_hhmm(sol['van_out'])} e deixa o grupo em FCO às "
           f"{min_to_hhmm(sol['van_drop'])}. Custo = {fmt_num(sol['cost'], 0)} EUR; "
           f"espera = {fmt_num(sol['wait'], 0)} min; $F$ = {fmt_num(sol['fitness'], 0)}.")
    write_latex_table(path,
                      ["Origem", "Ida", "EUR", "Esp.", "Volta", "EUR", "Esp."],
                      rows, "@{}lrrrrrr@{}",
                      f"Melhor roteiro encontrado pelo AG -- cenário {sc.key} ({sc.name}). "
                      "Esperas em minutos.",
                      f"tab:a1_best_{sc.key}", notes=van, tabcolsep="3.5pt")


def table_sensitivity(path: Path, rows_data: list[dict]) -> None:
    rows, last = [], None
    for d in rows_data:
        if last is not None and d["factor"] != last:
            rows.append("midrule")
        val = d["value"]
        val_txt = CX_NAMES.get(val, None) or (fmt_num(val, 2) if isinstance(val, float) else str(val))
        if d["is_base"]:
            val_txt = f"\\textbf{{{val_txt}}}"
        rows.append([d["factor"] if d["factor"] != last else "", val_txt,
                     fmt_num(d["mean"], 1), fmt_num(d["std"], 1),
                     f"{d['success']}/{d['runs']}",
                     fmt_num(d["hit_gen"], 0) if d["hit_gen"] is not None else "--"])
        last = d["factor"]
    write_latex_table(path, ["Fator", "Valor", "Média $F$", "DP", "Sucesso", "Ger."], rows,
                      "@{}llrrrr@{}",
                      "Análise de sensibilidade (um fator por vez) no cenário A. "
                      "Em negrito, o valor da configuração base.",
                      "tab:a1_sens",
                      notes="Sucesso: execuções que atingiram o ótimo exato. "
                            "Ger.: geração média em que $F^*$ foi atingido.")


# --------------------------------------------------------------------------
# Experimentos
# --------------------------------------------------------------------------

def sensitivity_study(table: FlightTable, sc: Scenario, base: GAParams, runs: int, seed: int,
                      f_star: float) -> list[dict]:
    cache: dict[GAParams, dict] = {}
    out = []
    for label, field, values in SENSITIVITY:
        for v in values:
            p = replace(base, **{field: v})
            if p not in cache:
                res = run_batch(table, sc, p, runs, seed)
                cache[p] = batch_metrics(res, f_star)
            m = cache[p]
            out.append({
                "factor": label, "field": field, "value": v, "is_base": p == base,
                "mean": m["best"]["mean"], "std": m["best"]["std"], "var": m["best"]["var"],
                "success": m["success"], "runs": m["runs"],
                "hit_gen": m["hit_gen"]["mean"] if m["hit_gen"] else None,
            })
            print(f"  [sens] {field}={v}: media={m['best']['mean']:.1f} "
                  f"sucesso={m['success']}/{m['runs']}", flush=True)
    return out


def tradeoff_study(table: FlightTable, sc: Scenario, sides, base: GAParams, runs: int,
                   seed: int, n_sample: int = 4000) -> dict:
    """Varre lambda e compara soluções do AG com a fronteira de Pareto exata."""
    out_s, ret_s = sides
    rng = np.random.default_rng(seed)
    fo = np.flatnonzero(out_s.ev.n_viol == 0)
    fr = np.flatnonzero(ret_s.ev.n_viol == 0)
    io, ir = rng.choice(fo, n_sample), rng.choice(fr, n_sample)
    sample = np.column_stack([out_s.ev.cost[io] + ret_s.ev.cost[ir],
                              out_s.ev.wait[io] + ret_s.ev.wait[ir]]).astype(float)
    front = pareto_front(table, sc, sides)
    ga_pts, ex_pts, rows = [], [], []
    for lam in LAMBDAS:
        s_l = sc.with_weight(lam)
        ex = exact_optimum(table, s_l, sides)
        d = describe_solution(table, s_l, ex.individual)
        ex_pts.append((lam, d["cost"], d["wait"]))
        res = run_batch(table, s_l, base, runs, seed)
        hits = 0
        for r in res:
            dr = describe_solution(table, s_l, r.best_individual)
            ga_pts.append((lam, dr["cost"], dr["wait"]))
            hits += r.first_hit(ex.fitness) is not None
        ga_f = np.array([r.best_fitness for r in res])
        # gap relativo ao ótimo de F_lambda (para lambda=0, F* é o custo mínimo)
        gap = 100.0 * (ga_f - ex.fitness) / ex.fitness
        rows.append({"scenario": sc.key, "lambda": lam, "exact_F": ex.fitness,
                     "exact_cost": d["cost"], "exact_wait": d["wait"],
                     "ga_mean_F": ga_f.mean(), "ga_mean_gap_pct": gap.mean(),
                     "ga_max_gap_pct": gap.max(), "ga_success": hits, "runs": runs})
    return {"key": sc.key, "sample": sample, "front": front, "ga": ga_pts, "exact": ex_pts,
            "rows": rows}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "flights.txt")
    ap.add_argument("--out", type=Path, default=None,
                    help="pasta de saída (padrão: results/atividade1; com --quick: results/quick/atividade1)")
    ap.add_argument("--seed", type=int, default=2026, help="seed base (execução r usa seed+r)")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="versão reduzida para teste rápido")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--skip-tradeoff", action="store_true")
    args = ap.parse_args(argv)

    if args.out is None:
        args.out = ROOT / "results" / ("quick" if args.quick else "") / "atividade1"
    plotstyle.apply_style()
    out, fig_dir, tab_dir = args.out, args.out / "figures", args.out / "tables"
    for d in (out, fig_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)

    base = BASE_PARAMS
    runs = args.runs
    sens_runs, trade_runs = runs, 10
    if args.quick:
        base = replace(base, generations=60)
        runs, sens_runs, trade_runs = 5, 3, 2

    table = load_flights(args.data)
    print(f"Espaco de busca: {table.search_space_size:.3e} cromossomos | genes: {table.gene_names()}")
    print(f"Parametros: {base.as_dict()}")

    summary = {"params": base.as_dict(), "seed": args.seed, "runs": runs, "scenarios": {}}
    all_results, optimum, ga_best, rs_best, sides_cache = {}, {}, {}, {}, {}

    for key, sc in SCENARIOS.items():
        print(f"\n== {sc.describe()}")
        t0 = time.perf_counter()
        sides = enumerate_sides(table, sc)
        sides_cache[key] = sides
        ex = exact_optimum(table, sc, sides)
        t_enum = time.perf_counter() - t0
        print(f"Otimo exato F* = {ex.fitness:.1f} ({ex.n_optima} cromossomo(s); "
              f"enumeracao de 2x10^6 combinacoes em {t_enum:.2f} s)")

        results = run_batch(table, sc, base, runs, args.seed)
        metrics = batch_metrics(results, ex.fitness)
        fit = make_fitness(table, sc)
        rs = [random_search(fit, table.gene_upper, base.evaluations, seed=args.seed + 10_000 + r)[1]
              for r in range(runs)]

        best_run = min(results, key=lambda r: r.best_fitness)
        sol = describe_solution(table, sc, best_run.best_individual)
        txt = format_solution(sol)
        print(txt)
        print(f"AG: media={metrics['best']['mean']:.1f} var={metrics['best']['var']:.1f} "
              f"dp={metrics['best']['std']:.1f} sucesso={metrics['success']}/{runs} | "
              f"busca aleatoria: media={np.mean(rs):.1f}")

        (out / f"melhor_solucao_{key}.txt").write_text(sc.describe() + "\n" + txt + "\n", encoding="utf-8")
        save_csv(out / f"execucoes_{key}.csv", [{
            "run": i, "seed": r.seed, "best_fitness": r.best_fitness,
            "hit_generation": r.first_hit(ex.fitness), "final_pop_mean": r.final_fitness.mean(),
            "final_pop_std": r.final_fitness.std(ddof=1), "elapsed_s": r.elapsed_s,
            "genes": " ".join(map(str, r.best_individual)), "random_search_best": rs[i],
        } for i, r in enumerate(results)])
        np.savez_compressed(out / f"historicos_{key}.npz",
                            **{m: np.vstack([r.history[m] for r in results])
                               for m in ("best", "mean", "worst", "std", "unique")})
        table_schedule(tab_dir / f"tab_a1_roteiro_{key}.tex", sol, sc)
        plots.plot_schedule(sol, sc, fig_dir, f"a1_roteiro_{key}")

        summary["scenarios"][key] = {
            "description": sc.describe(),
            "exact": {"fitness": ex.fitness, "genes": ex.individual, "n_optima": ex.n_optima,
                      "enum_time_s": t_enum},
            "ga": metrics,
            "random": summarize(rs),
            "best_solution": sol,
        }
        all_results[key], optimum[key] = results, ex.fitness
        ga_best[key] = [r.best_fitness for r in results]
        rs_best[key] = rs

    plots.plot_convergence(all_results, optimum, SCENARIOS, fig_dir, "a1_convergencia")
    plots.plot_final_distribution(ga_best, rs_best, optimum, fig_dir, "a1_distribuicao_final")
    table_params(tab_dir / "tab_a1_parametros.tex", base, runs, args.seed)
    table_stats(tab_dir / "tab_a1_estatisticas.tex", summary)

    if not args.skip_sensitivity:
        print("\n== Analise de sensibilidade (cenario A)")
        sens = sensitivity_study(table, SCENARIOS["A"], base, sens_runs, args.seed, optimum["A"])
        save_csv(out / "sensibilidade_A.csv", sens)
        table_sensitivity(tab_dir / "tab_a1_sensibilidade.tex", sens)
        summary["sensitivity"] = sens

    if not args.skip_tradeoff:
        print("\n== Varredura de lambda (custo x espera)")
        panels = [tradeoff_study(table, sc, sides_cache[k], base, trade_runs, args.seed)
                  for k, sc in SCENARIOS.items()]
        plots.plot_tradeoff(panels, fig_dir, "a1_custo_vs_espera")
        rows = [r for p in panels for r in p["rows"]]
        save_csv(out / "tradeoff_lambda.csv", rows)
        summary["tradeoff"] = rows
        summary["pareto_front"] = {p["key"]: p["front"] for p in panels}
        for r in rows:
            print(f"  {r['scenario']} lambda={r['lambda']}: C={r['exact_cost']:.0f} "
                  f"W={r['exact_wait']:.0f} sucesso AG={r['ga_success']}/{r['runs']} "
                  f"gap medio={r['ga_mean_gap_pct']:.2f}% gap max={r['ga_max_gap_pct']:.2f}%")

    save_json(out / "resumo.json", summary)
    print(f"\nResultados salvos em {out}")
    return summary


if __name__ == "__main__":
    main()

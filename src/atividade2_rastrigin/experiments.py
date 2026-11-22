"""Experimentos da Atividade 2 (Rastrigin com DEAP).

Uso (a partir da raiz do projeto)::

    python -m src.atividade2_rastrigin.experiments                 # completo
    python -m src.atividade2_rastrigin.experiments --quick         # teste rápido
    python -m src.atividade2_rastrigin.experiments --jobs 4        # execuções em paralelo

As execuções são independentes e cada uma usa a seed ``seed + r``; portanto o
paralelismo (``--jobs``) não altera os resultados.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..common import plotstyle
from ..common.stats import fmt_num, fmt_sci, save_csv, save_json, summarize, write_latex_table
from . import plots
from .ga_deap import DeapParams, DeapResult, random_search, run_deap

ROOT = Path(__file__).resolve().parents[2]

EPS_SUCCESS = 1e-4   # sucesso: f(x_best) < 1e-4 (ótimo global com tolerância)
BASIN = 0.5          # f < 0,5  <=>  todas as coordenadas na bacia do ótimo global

BASE_PARAMS = {
    2: DeapParams(n_dim=2, pop_size=100, generations=100, tournament_size=3, cxpb=0.9,
                  mutpb=1.0, indpb=None, eta_c=15.0, eta_m=20.0, elitism=2),
    10: DeapParams(n_dim=10, pop_size=200, generations=500, tournament_size=3, cxpb=0.9,
                   mutpb=1.0, indpb=None, eta_c=15.0, eta_m=20.0, elitism=2),
}
SNAPSHOT_GENS = (0, 5, 15, 100)

SENSITIVITY = [
    ("Torneio $k$", "tournament_size", [2, 3, 5]),
    ("$p_c$", "cxpb", [0.6, 0.9, 1.0]),
    ("$p_{\\mathrm{gene}}$", "indpb", [0.05, 0.1, 0.2]),
    ("$\\eta_m$", "eta_m", [5.0, 20.0, 50.0]),
    ("Elitismo $e$", "elitism", [0, 2, 10]),
]


def _key(p: DeapParams) -> tuple:
    return tuple(sorted(p.as_dict().items()))


def _run_one(args) -> DeapResult:
    p, seed, snaps = args
    return run_deap(p, seed=seed, snapshot_gens=snaps)


def run_batch(p: DeapParams, runs: int, seed: int, jobs: int = 1,
              snapshot_gens: tuple[int, ...] = ()) -> list[DeapResult]:
    tasks = [(p, seed + r, snapshot_gens if r == 0 else ()) for r in range(runs)]
    if jobs <= 1:
        return [_run_one(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(_run_one, tasks))


def batch_metrics(res: list[DeapResult]) -> dict:
    best = np.array([r.best_fitness for r in res])
    hits = [r.first_hit(EPS_SUCCESS) for r in res]
    hit_gens = [h for h in hits if h is not None]
    basin_gens = [h for h in (r.first_hit(BASIN) for r in res) if h is not None]
    return {
        "best": summarize(best),
        "log10_best": summarize(np.log10(np.maximum(best, 1e-300))),
        "success": len(hit_gens),
        "basin": int((best < BASIN).sum()),
        "runs": len(res),
        "hit_gen": summarize(hit_gens) if hit_gens else None,
        "basin_gen": summarize(basin_gens) if basin_gens else None,
        "dist_origin": summarize([np.linalg.norm(r.best) for r in res]),
        "final_pop_mean": summarize([r.history["mean"][-1] for r in res]),
        "final_pop_std": summarize([r.history["std"][-1] for r in res]),
        "evaluations": summarize([r.evaluations for r in res]),
        "time_s": summarize([r.elapsed_s for r in res]),
    }


# --------------------------------------------------------------------------
# Tabelas
# --------------------------------------------------------------------------

def table_params(path: Path, params: dict[int, DeapParams], runs: int, seed: int) -> None:
    dims = list(params)
    P = [params[n] for n in dims]

    def row(label, fn):
        return [label] + [fn(p) for p in P]

    rows = [
        row("Representação", lambda p: f"{p.n_dim} reais"),
        row("População $N$", lambda p: str(p.pop_size)),
        row("Gerações $G$", lambda p: str(p.generations)),
        row("Seleção (torneio)", lambda p: f"$k={p.tournament_size}$"),
        row("Cruzamento SBX: $p_c$ / $\\eta_c$", lambda p: f"{fmt_num(p.cxpb, 1)} / {p.eta_c:g}"),
        row("Mutação polinomial: $\\eta_m$", lambda p: f"{p.eta_m:g}"),
        row("\\quad prob. por indivíduo / gene", lambda p: f"{fmt_num(p.mutpb, 1)} / "
                                                         f"{fmt_num(p.gene_mut_prob, 2)}"),
        row("Elitismo", lambda p: f"{p.elitism} melhores"),
        row("Avaliações/execução (máx.)", lambda p: fmt_num(p.evaluations_upper_bound, 0)),
        row("Execuções", lambda p: str(runs)),
    ]
    write_latex_table(path, ["Parâmetro"] + [f"$n={n}$" for n in dims], rows,
                      "@{}l" + "c" * len(dims) + "@{}",
                      f"Parâmetros do AG da Atividade 2 (DEAP); domínio $[-5{{,}}12;\\,5{{,}}12]^n$, "
                      f"seeds {seed}--{seed + runs - 1}.", "tab:a2_params")


def table_stats(path: Path, summary: dict) -> None:
    dims = list(summary["dims"])
    S = summary["dims"]

    def row(label, fn):
        return [label] + [fn(S[n]) for n in dims]

    rows = [
        row("Média do melhor $f$", lambda s: fmt_sci(s["ga"]["best"]["mean"])),
        row("Variância", lambda s: fmt_sci(s["ga"]["best"]["var"])),
        row("Desvio padrão", lambda s: fmt_sci(s["ga"]["best"]["std"])),
        row("Mínimo", lambda s: fmt_sci(s["ga"]["best"]["min"])),
        row("Mediana", lambda s: fmt_sci(s["ga"]["best"]["median"])),
        row("Máximo", lambda s: fmt_sci(s["ga"]["best"]["max"])),
        row("Média de $\\log_{10} f$ ($\\pm$ DP)", lambda s:
            f"${fmt_num(s['ga']['log10_best']['mean'], 2)} \\pm {fmt_num(s['ga']['log10_best']['std'], 2)}$"),
        row("$\\|\\mathbf{x}_{best}\\|$ médio", lambda s: fmt_sci(s["ga"]["dist_origin"]["mean"])),
        row(f"Execuções com $f<0{{,}}5$ (bacia global)", lambda s: f"{s['ga']['basin']}/{s['ga']['runs']}"),
        row(f"Execuções com $f<10^{{-4}}$", lambda s: f"{s['ga']['success']}/{s['ga']['runs']}"),
        row("Geração p/ $f<10^{-4}$ (média)", lambda s: fmt_num(s["ga"]["hit_gen"]["mean"], 1)
            if s["ga"]["hit_gen"] else "--"),
        row("$\\bar f$ da pop. final (média)", lambda s: fmt_num(s["ga"]["final_pop_mean"]["mean"], 3)),
        row("Tempo por execução (s)", lambda s: fmt_num(s["ga"]["time_s"]["mean"], 2)),
        "midrule",
        row("Busca aleatória: média", lambda s: fmt_num(s["random"]["mean"], 3)),
        row("Busca aleatória: desvio padrão", lambda s: fmt_num(s["random"]["std"], 3)),
    ]
    write_latex_table(path, ["Métrica (30 execuções)"] + [f"$n={n}$" for n in dims], rows,
                      "@{}l" + "r" * len(dims) + "@{}",
                      "Estatísticas do melhor valor final de $f$ na Atividade 2.",
                      "tab:a2_stats",
                      notes="Variância e desvio padrão amostrais. A busca aleatória usa o "
                            "número máximo de avaliações do AG.")


def table_sensitivity(path: Path, rows_data: list[dict]) -> None:
    rows, last = [], None
    for d in rows_data:
        if last is not None and d["factor"] != last:
            rows.append("midrule")
        v = d["value"]
        txt = fmt_num(v, 2) if isinstance(v, float) else str(v)
        if d["is_base"]:
            txt = f"\\textbf{{{txt}}}"
        rows.append([d["factor"] if d["factor"] != last else "", txt,
                     fmt_sci(d["mean"]), fmt_sci(d["median"]),
                     f"{d['basin']}/{d['runs']}", f"{d['success']}/{d['runs']}"])
        last = d["factor"]
    write_latex_table(path, ["Fator", "Valor", "Média $f$", "Mediana $f$", "$f<0{,}5$", "$f<10^{-4}$"],
                      rows, "@{}llrrrr@{}",
                      "Sensibilidade dos parâmetros do AG/DEAP na Rastrigin com $n=10$ "
                      "(um fator por vez; base em negrito).", "tab:a2_sens", tabcolsep="3pt")


# --------------------------------------------------------------------------

def sensitivity_study(base: DeapParams, runs: int, seed: int, jobs: int) -> list[dict]:
    cache: dict[tuple, dict] = {}
    out = []
    for label, field, values in SENSITIVITY:
        for v in values:
            p = replace(base, **{field: v})
            k = _key(p)
            if k not in cache:
                cache[k] = batch_metrics(run_batch(p, runs, seed, jobs))
            m = cache[k]
            out.append({"factor": label, "field": field, "value": v, "is_base": k == _key(base),
                        "mean": m["best"]["mean"], "median": m["best"]["median"],
                        "std": m["best"]["std"], "var": m["best"]["var"],
                        "basin": m["basin"], "success": m["success"], "runs": m["runs"]})
            print(f"  [sens] {field}={v}: media={m['best']['mean']:.3g} "
                  f"bacia={m['basin']}/{runs} sucesso={m['success']}/{runs}", flush=True)
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="pasta de saída (padrão: results/atividade2; com --quick: results/quick/atividade2)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 10])
    ap.add_argument("--jobs", type=int, default=1, help="processos em paralelo")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args(argv)

    if args.out is None:
        args.out = ROOT / "results" / ("quick" if args.quick else "") / "atividade2"
    plotstyle.apply_style()
    out, fig_dir, tab_dir = args.out, args.out / "figures", args.out / "tables"
    for d in (out, fig_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)

    params = {n: BASE_PARAMS.get(n, replace(BASE_PARAMS[10], n_dim=n)) for n in args.dims}
    runs, sens_runs = args.runs, args.runs
    if args.quick:
        params = {n: replace(p, generations=min(p.generations, 60)) for n, p in params.items()}
        runs, sens_runs = 4, 3

    summary = {"seed": args.seed, "runs": runs, "eps_success": EPS_SUCCESS, "dims": {}}
    results, ga_best, rs_best = {}, {}, {}
    for n, p in params.items():
        print(f"\n== Rastrigin n={n} | {p.as_dict()}")
        snaps = SNAPSHOT_GENS if n == 2 else ()
        res = run_batch(p, runs, args.seed, args.jobs, snapshot_gens=snaps)
        m = batch_metrics(res)
        rs = [random_search(n, p.evaluations_upper_bound, seed=args.seed + 10_000 + r)
              for r in range(runs)]
        best_run = min(res, key=lambda r: r.best_fitness)
        print(f"AG: media={m['best']['mean']:.3e} var={m['best']['var']:.3e} dp={m['best']['std']:.3e} "
              f"bacia={m['basin']}/{runs} sucesso={m['success']}/{runs} | "
              f"busca aleatoria: media={np.mean(rs):.3f}")
        print(f"Melhor x = {np.round(best_run.best, 6).tolist()} f = {best_run.best_fitness:.3e}")

        save_csv(out / f"execucoes_n{n}.csv", [{
            "run": i, "seed": r.seed, "best_f": r.best_fitness,
            "gen_f_lt_1e-4": r.first_hit(EPS_SUCCESS), "gen_f_lt_0.5": r.first_hit(BASIN),
            "norm_x": float(np.linalg.norm(r.best)), "evaluations": r.evaluations,
            "elapsed_s": r.elapsed_s, "random_search_best": rs[i],
            "x_best": " ".join(f"{v:.6g}" for v in r.best),
        } for i, r in enumerate(res)])
        np.savez_compressed(out / f"historicos_n{n}.npz",
                            **{k: np.vstack([r.history[k] for r in res])
                               for k in ("best", "mean", "worst", "std")})
        # logbook da primeira execução (formato texto do DEAP)
        (out / f"logbook_n{n}_seed{args.seed}.txt").write_text(str(res[0].logbook), encoding="utf-8")

        summary["dims"][n] = {"params": p.as_dict(), "ga": m, "random": summarize(rs),
                              "best_x": best_run.best, "best_f": best_run.best_fitness}
        results[n], ga_best[n], rs_best[n] = res, [r.best_fitness for r in res], rs

        if n == 2:
            pts = np.array([r.best for r in res])
            plots.plot_surface(pts, fig_dir, "a2_superficie")
            if res[0].snapshots:
                plots.plot_population_snapshots(res[0].snapshots, fig_dir, "a2_populacao_n2")

    plots.plot_convergence(results, fig_dir, "a2_convergencia")
    plots.plot_final_distribution(ga_best, rs_best, fig_dir, "a2_distribuicao_final")
    table_params(tab_dir / "tab_a2_parametros.tex", params, runs, args.seed)
    table_stats(tab_dir / "tab_a2_estatisticas.tex", summary)

    if not args.skip_sensitivity and 10 in params:
        print("\n== Analise de sensibilidade (n=10)")
        sens = sensitivity_study(params[10], sens_runs, args.seed, args.jobs)
        save_csv(out / "sensibilidade_n10.csv", sens)
        table_sensitivity(tab_dir / "tab_a2_sensibilidade.tex", sens)
        summary["sensitivity"] = sens

    save_json(out / "resumo.json", summary)
    print(f"\nResultados salvos em {out}")
    return summary


if __name__ == "__main__":
    main()

"""Uso (a partir da raiz do projeto):

    python -m src.atividade2_rastrigin.experimentos
    python -m src.atividade2_rastrigin.experimentos --rapido
    python -m src.atividade2_rastrigin.experimentos --processos 4 --dimensoes 2 10
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..comum import estilo_graficos
from ..comum.estatistica import (escrever_tabela_latex, formatar_cientifico, formatar_num, resumir,
                                 salvar_csv, salvar_json)
from . import graficos
from .ag_deap import ParametrosDeap, ResultadoDeap, busca_aleatoria, executar_deap

RAIZ = Path(__file__).resolve().parents[2]

TOL_SUCESSO = 1e-4
LIMIAR_BACIA = 0.5

PARAMETROS_BASE = {
    2: ParametrosDeap(dimensao=2, tam_populacao=100, geracoes=100, tam_torneio=3, prob_cruzamento=0.9,
                      prob_mutacao=1.0, prob_gene=None, eta_c=15.0, eta_m=20.0, elitismo=2),
    10: ParametrosDeap(dimensao=10, tam_populacao=200, geracoes=500, tam_torneio=3, prob_cruzamento=0.9,
                       prob_mutacao=1.0, prob_gene=None, eta_c=15.0, eta_m=20.0, elitismo=2),
}
GERACOES_RETRATO = (0, 5, 15, 100)

SENSIBILIDADE = [
    ("Torneio $k$", "tam_torneio", [2, 3, 5]),
    ("$p_c$", "prob_cruzamento", [0.6, 0.9, 1.0]),
    ("$p_{\\mathrm{gene}}$", "prob_gene", [0.05, 0.1, 0.2]),
    ("$\\eta_m$", "eta_m", [5.0, 20.0, 50.0]),
    ("Elitismo $e$", "elitismo", [0, 2, 10]),
]


def _chave(p: ParametrosDeap) -> tuple:
    return tuple(sorted(p.como_dict().items()))


def _executar_uma(args) -> ResultadoDeap:
    p, seed, retratos = args
    return executar_deap(p, seed=seed, geracoes_retrato=retratos)


def executar_lote(p: ParametrosDeap, execucoes: int, seed: int, processos: int = 1,
                  geracoes_retrato: tuple[int, ...] = ()) -> list[ResultadoDeap]:
    tarefas = [(p, seed + r, geracoes_retrato if r == 0 else ()) for r in range(execucoes)]
    if processos <= 1:
        return [_executar_uma(t) for t in tarefas]
    with ProcessPoolExecutor(max_workers=processos) as executor:
        return list(executor.map(_executar_uma, tarefas))


def metricas_lote(resultados: list[ResultadoDeap]) -> dict:
    melhores = np.array([r.melhor_fitness for r in resultados])
    ger_sucesso = [g for g in (r.geracao_abaixo_de(TOL_SUCESSO) for r in resultados) if g is not None]
    ger_bacia = [g for g in (r.geracao_abaixo_de(LIMIAR_BACIA) for r in resultados) if g is not None]
    return {
        "melhor": resumir(melhores),
        "log10_melhor": resumir(np.log10(np.maximum(melhores, 1e-300))),
        "sucessos": len(ger_sucesso),
        "na_bacia": int((melhores < LIMIAR_BACIA).sum()),
        "execucoes": len(resultados),
        "geracao_sucesso": resumir(ger_sucesso) if ger_sucesso else None,
        "geracao_bacia": resumir(ger_bacia) if ger_bacia else None,
        "distancia_origem": resumir([np.linalg.norm(r.melhor) for r in resultados]),
        "media_pop_final": resumir([r.historico["medio"][-1] for r in resultados]),
        "desvio_pop_final": resumir([r.historico["desvio"][-1] for r in resultados]),
        "avaliacoes": resumir([r.avaliacoes for r in resultados]),
        "tempo_s": resumir([r.tempo_s for r in resultados]),
    }


def tabela_parametros(caminho: Path, params: dict[int, ParametrosDeap], execucoes: int, seed: int) -> None:
    dimensoes = list(params)
    P = [params[n] for n in dimensoes]

    def linha(rotulo, fn):
        return [rotulo] + [fn(p) for p in P]

    linhas = [
        linha("Representação", lambda p: f"{p.dimensao} reais"),
        linha("População $N$", lambda p: str(p.tam_populacao)),
        linha("Gerações $G$", lambda p: str(p.geracoes)),
        linha("Seleção (torneio)", lambda p: f"$k={p.tam_torneio}$"),
        linha("Cruzamento SBX: $p_c$ / $\\eta_c$",
              lambda p: f"{formatar_num(p.prob_cruzamento, 1)} / {p.eta_c:g}"),
        linha("Mutação polinomial: $\\eta_m$", lambda p: f"{p.eta_m:g}"),
        linha("\\quad prob. por indivíduo / gene",
              lambda p: f"{formatar_num(p.prob_mutacao, 1)} / {formatar_num(p.prob_mutacao_gene, 2)}"),
        linha("Elitismo", lambda p: f"{p.elitismo} melhores"),
        linha("Avaliações/execução (máx.)", lambda p: formatar_num(p.limite_avaliacoes, 0)),
        linha("Execuções", lambda p: str(execucoes)),
    ]
    escrever_tabela_latex(caminho, ["Parâmetro"] + [f"$n={n}$" for n in dimensoes], linhas,
                          "@{}l" + "c" * len(dimensoes) + "@{}",
                          f"Parâmetros do AG da Atividade 2 (DEAP); domínio $[-5{{,}}12;\\,5{{,}}12]^n$, "
                          f"seeds {seed}--{seed + execucoes - 1}.", "tab:a2_params")


def tabela_estatisticas(caminho: Path, resumo: dict) -> None:
    dimensoes = list(resumo["dimensoes"])
    S = resumo["dimensoes"]

    def linha(rotulo, fn):
        return [rotulo] + [fn(S[n]) for n in dimensoes]

    linhas = [
        linha("Média do melhor $f$", lambda s: formatar_cientifico(s["ag"]["melhor"]["media"])),
        linha("Variância", lambda s: formatar_cientifico(s["ag"]["melhor"]["variancia"])),
        linha("Desvio padrão", lambda s: formatar_cientifico(s["ag"]["melhor"]["desvio_padrao"])),
        linha("Mínimo", lambda s: formatar_cientifico(s["ag"]["melhor"]["minimo"])),
        linha("Mediana", lambda s: formatar_cientifico(s["ag"]["melhor"]["mediana"])),
        linha("Máximo", lambda s: formatar_cientifico(s["ag"]["melhor"]["maximo"])),
        linha("Média de $\\log_{10} f$ ($\\pm$ DP)", lambda s:
              f"${formatar_num(s['ag']['log10_melhor']['media'], 2)} \\pm "
              f"{formatar_num(s['ag']['log10_melhor']['desvio_padrao'], 2)}$"),
        linha("$\\|\\mathbf{x}_{best}\\|$ médio",
              lambda s: formatar_cientifico(s["ag"]["distancia_origem"]["media"])),
        linha("Execuções com $f<0{,}5$ (bacia global)",
              lambda s: f"{s['ag']['na_bacia']}/{s['ag']['execucoes']}"),
        linha("Execuções com $f<10^{-4}$", lambda s: f"{s['ag']['sucessos']}/{s['ag']['execucoes']}"),
        linha("Geração p/ $f<10^{-4}$ (média)",
              lambda s: formatar_num(s["ag"]["geracao_sucesso"]["media"], 1)
              if s["ag"]["geracao_sucesso"] else "--"),
        linha("$\\bar f$ da pop. final (média)",
              lambda s: formatar_num(s["ag"]["media_pop_final"]["media"], 3)),
        linha("Tempo por execução (s)", lambda s: formatar_num(s["ag"]["tempo_s"]["media"], 2)),
        "midrule",
        linha("Busca aleatória: média", lambda s: formatar_num(s["aleatoria"]["media"], 3)),
        linha("Busca aleatória: desvio padrão", lambda s: formatar_num(s["aleatoria"]["desvio_padrao"], 3)),
    ]
    escrever_tabela_latex(caminho, [f"Métrica ({resumo['execucoes']} execuções)"]
                          + [f"$n={n}$" for n in dimensoes], linhas,
                          "@{}l" + "r" * len(dimensoes) + "@{}",
                          "Estatísticas do melhor valor final de $f$ na Atividade 2.",
                          "tab:a2_stats",
                          notas="Variância e desvio padrão amostrais. A busca aleatória usa o "
                                "número máximo de avaliações do AG.")


def tabela_sensibilidade(caminho: Path, dados: list[dict]) -> None:
    linhas, anterior = [], None
    for d in dados:
        if anterior is not None and d["fator"] != anterior:
            linhas.append("midrule")
        v = d["valor"]
        texto = formatar_num(v, 2) if isinstance(v, float) else str(v)
        if d["eh_base"]:
            texto = f"\\textbf{{{texto}}}"
        linhas.append([d["fator"] if d["fator"] != anterior else "", texto,
                       formatar_cientifico(d["media"]), formatar_cientifico(d["mediana"]),
                       f"{d['na_bacia']}/{d['execucoes']}", f"{d['sucessos']}/{d['execucoes']}"])
        anterior = d["fator"]
    escrever_tabela_latex(caminho, ["Fator", "Valor", "Média $f$", "Mediana $f$", "$f<0{,}5$",
                                    "$f<10^{-4}$"],
                          linhas, "@{}llrrrr@{}",
                          "Sensibilidade dos parâmetros do AG/DEAP na Rastrigin com $n=10$ "
                          "(um fator por vez; base em negrito).", "tab:a2_sens", espaco_colunas="3pt")


def estudo_sensibilidade(base: ParametrosDeap, execucoes: int, seed: int, processos: int) -> list[dict]:
    cache: dict[tuple, dict] = {}
    saida = []
    for rotulo, campo, valores in SENSIBILIDADE:
        for v in valores:
            p = replace(base, **{campo: v})
            k = _chave(p)
            if k not in cache:
                cache[k] = metricas_lote(executar_lote(p, execucoes, seed, processos))
            m = cache[k]
            saida.append({"fator": rotulo, "campo": campo, "valor": v, "eh_base": k == _chave(base),
                          "media": m["melhor"]["media"], "mediana": m["melhor"]["mediana"],
                          "desvio_padrao": m["melhor"]["desvio_padrao"],
                          "variancia": m["melhor"]["variancia"],
                          "na_bacia": m["na_bacia"], "sucessos": m["sucessos"],
                          "execucoes": m["execucoes"]})
            print(f"  [sens] {campo}={v}: media={m['melhor']['media']:.3g} "
                  f"bacia={m['na_bacia']}/{execucoes} sucesso={m['sucessos']}/{execucoes}", flush=True)
    return saida


def principal(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", "--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--execucoes", "--runs", type=int, default=30)
    ap.add_argument("--dimensoes", "--dims", type=int, nargs="+", default=[2, 10])
    ap.add_argument("--processos", "--jobs", type=int, default=1)
    ap.add_argument("--rapido", "--quick", action="store_true")
    ap.add_argument("--sem-sensibilidade", "--skip-sensitivity", action="store_true")
    args = ap.parse_args(argv)

    if args.saida is None:
        args.saida = RAIZ / "results" / ("quick" if args.rapido else "") / "atividade2"
    estilo_graficos.aplicar_estilo()
    saida, pasta_fig, pasta_tab = args.saida, args.saida / "figures", args.saida / "tables"
    for d in (saida, pasta_fig, pasta_tab):
        d.mkdir(parents=True, exist_ok=True)

    params = {n: PARAMETROS_BASE.get(n, replace(PARAMETROS_BASE[10], dimensao=n)) for n in args.dimensoes}
    execucoes, exec_sens = args.execucoes, args.execucoes
    if args.rapido:
        params = {n: replace(p, geracoes=min(p.geracoes, 60)) for n, p in params.items()}
        execucoes, exec_sens = 4, 3

    resumo = {"seed": args.seed, "execucoes": execucoes, "tol_sucesso": TOL_SUCESSO, "dimensoes": {}}
    resultados, melhores_ag, melhores_aleat = {}, {}, {}
    for n, p in params.items():
        print(f"\n== Rastrigin n={n} | {p.como_dict()}")
        retratos = GERACOES_RETRATO if n == 2 else ()
        res = executar_lote(p, execucoes, args.seed, args.processos, geracoes_retrato=retratos)
        m = metricas_lote(res)
        aleat = [busca_aleatoria(n, p.limite_avaliacoes, seed=args.seed + 10_000 + r)
                 for r in range(execucoes)]
        melhor_exec = min(res, key=lambda r: r.melhor_fitness)
        print(f"AG: media={m['melhor']['media']:.3e} var={m['melhor']['variancia']:.3e} "
              f"dp={m['melhor']['desvio_padrao']:.3e} bacia={m['na_bacia']}/{execucoes} "
              f"sucesso={m['sucessos']}/{execucoes} | busca aleatoria: media={np.mean(aleat):.3f}")
        print(f"Melhor x = {np.round(melhor_exec.melhor, 6).tolist()} f = {melhor_exec.melhor_fitness:.3e}")

        salvar_csv(saida / f"execucoes_n{n}.csv", [{
            "execucao": i, "seed": r.seed, "melhor_f": r.melhor_fitness,
            "geracao_f_menor_1e-4": r.geracao_abaixo_de(TOL_SUCESSO),
            "geracao_f_menor_0.5": r.geracao_abaixo_de(LIMIAR_BACIA),
            "norma_x": float(np.linalg.norm(r.melhor)), "avaliacoes": r.avaliacoes,
            "tempo_s": r.tempo_s, "melhor_busca_aleatoria": aleat[i],
            "melhor_x": " ".join(f"{v:.6g}" for v in r.melhor),
        } for i, r in enumerate(res)])
        np.savez_compressed(saida / f"historicos_n{n}.npz",
                            **{k: np.vstack([r.historico[k] for r in res])
                               for k in ("melhor", "medio", "pior", "desvio")})
        (saida / f"logbook_n{n}_seed{args.seed}.txt").write_text(str(res[0].logbook), encoding="utf-8")

        resumo["dimensoes"][n] = {"parametros": p.como_dict(), "ag": m, "aleatoria": resumir(aleat),
                                  "melhor_x": melhor_exec.melhor, "melhor_f": melhor_exec.melhor_fitness}
        resultados[n], melhores_ag[n], melhores_aleat[n] = res, [r.melhor_fitness for r in res], aleat

        if n == 2:
            pontos = np.array([r.melhor for r in res])
            graficos.grafico_superficie(pontos, pasta_fig, "a2_superficie")
            if res[0].retratos:
                graficos.grafico_populacao(res[0].retratos, pasta_fig, "a2_populacao_n2")

    graficos.grafico_convergencia(resultados, pasta_fig, "a2_convergencia")
    graficos.grafico_distribuicao_final(melhores_ag, melhores_aleat, pasta_fig, "a2_distribuicao_final")
    tabela_parametros(pasta_tab / "tab_a2_parametros.tex", params, execucoes, args.seed)
    tabela_estatisticas(pasta_tab / "tab_a2_estatisticas.tex", resumo)

    if not args.sem_sensibilidade and 10 in params:
        print("\n== Analise de sensibilidade (n=10)")
        sens = estudo_sensibilidade(params[10], exec_sens, args.seed, args.processos)
        salvar_csv(saida / "sensibilidade_n10.csv", sens)
        tabela_sensibilidade(pasta_tab / "tab_a2_sensibilidade.tex", sens)
        resumo["sensibilidade"] = sens

    salvar_json(saida / "resumo.json", resumo)
    print(f"\nResultados salvos em {saida}")
    return resumo


if __name__ == "__main__":
    principal()

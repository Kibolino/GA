"""Uso (a partir da raiz do projeto):

    python -m src.atividade1_voos.experimentos
    python -m src.atividade1_voos.experimentos --rapido
    python -m src.atividade1_voos.experimentos --seed 7 --execucoes 50
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..comum import estilo_graficos
from ..comum.estatistica import (escrever_tabela_latex, formatar_num, resumir, salvar_csv,
                                 salvar_json)
from . import graficos
from .ag import ParametrosAG, ResultadoAG, busca_aleatoria, executar_ag
from .exato import enumerar_sentidos, fronteira_pareto, otimo_exato
from .modelo import CENARIOS, Cenario, criar_fitness, descrever_solucao, formatar_solucao
from .voos import TabelaVoos, carregar_voos, min_para_hhmm

RAIZ = Path(__file__).resolve().parents[2]

PARAMETROS_BASE = ParametrosAG(
    tam_populacao=150,
    geracoes=400,
    tam_torneio=3,
    cruzamento="uniforme",
    prob_cruzamento=0.9,
    prob_mutacao=0.05,
    prob_creep=0.5,
    prob_deslocamento=0.3,
    elitismo=5,
)

SENSIBILIDADE = [
    ("Torneio $k$", "tam_torneio", [2, 3, 5]),
    ("Cruzamento", "cruzamento", ["um_ponto", "dois_pontos", "uniforme"]),
    ("$p_c$", "prob_cruzamento", [0.6, 0.9, 1.0]),
    ("$p_m$ (gene)", "prob_mutacao", [0.02, 0.05, 0.10]),
    ("$p_{\\mathrm{shift}}$", "prob_deslocamento", [0.0, 0.1, 0.3, 0.5]),
    ("Elitismo $e$", "elitismo", [0, 1, 5, 10]),
    ("População $N$", "tam_populacao", [50, 150, 300]),
]
NOMES_CRUZAMENTO = {"um_ponto": "1 ponto", "dois_pontos": "2 pontos", "uniforme": "uniforme"}
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def grupos_deslocamento(tabela: TabelaVoos) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(0, tabela.n_genes, 2), np.arange(1, tabela.n_genes, 2)


def executar_lote(tabela: TabelaVoos, cen: Cenario, params: ParametrosAG, execucoes: int,
                  seed: int) -> list[ResultadoAG]:
    fitness = criar_fitness(tabela, cen)
    grupos = grupos_deslocamento(tabela)
    return [executar_ag(fitness, tabela.limite_genes, params, seed=seed + r,
                        grupos_deslocamento=grupos)
            for r in range(execucoes)]


def metricas_lote(resultados: list[ResultadoAG], f_otimo: float) -> dict:
    melhores = [r.melhor_fitness for r in resultados]
    geracoes_acerto = [g for g in (r.geracao_que_atingiu(f_otimo) for r in resultados) if g is not None]
    return {
        "melhor": resumir(melhores),
        "gap_medio_pct": 100.0 * (np.mean(melhores) - f_otimo) / f_otimo,
        "sucessos": len(geracoes_acerto),
        "execucoes": len(resultados),
        "geracao_acerto": resumir(geracoes_acerto) if geracoes_acerto else None,
        "media_pop_final": resumir([r.fitness_final.mean() for r in resultados]),
        "desvio_pop_final": resumir([r.fitness_final.std(ddof=1) for r in resultados]),
        "tempo_s": resumir([r.tempo_s for r in resultados]),
    }


def tabela_parametros(caminho: Path, p: ParametrosAG, execucoes: int, seed: int) -> None:
    ca, cb = CENARIOS["A"], CENARIOS["B"]
    linhas = [
        ["Representação", "12 inteiros (ida/volta de cada pesquisador), gene $\\in\\{0..9\\}$"],
        ["Tamanho da população $N$", f"{p.tam_populacao}"],
        ["Número de gerações $G$", f"{p.geracoes}"],
        ["Seleção", f"torneio determinístico, $k={p.tam_torneio}$, com reposição"],
        ["Cruzamento", f"{NOMES_CRUZAMENTO[p.cruzamento]}, $p_c={formatar_num(p.prob_cruzamento, 2)}$"],
        ["Mutação por gene", f"$p_m={formatar_num(p.prob_mutacao, 2)}$ "
                             f"({int(p.prob_creep * 100)}\\% creep $\\pm1$, resto reset)"],
        ["Mutação de bloco", f"$p_{{\\mathrm{{shift}}}}={formatar_num(p.prob_deslocamento, 2)}$ por filho"],
        ["Elitismo", f"{p.elitismo} melhores copiados sem alteração"],
        ["Avaliações/execução", formatar_num(p.avaliacoes, 0)],
        ["Execuções", f"{execucoes} (seeds {seed}--{seed + execucoes - 1})"],
        "midrule",
        ["$\\lambda$ (custo da espera)", f"{formatar_num(ca.peso_espera, 0)} EUR/min"],
        ["Check-in / translado", f"{ca.checkin_min} min / {ca.translado_min} min"],
        ["Janela (cenário B)", f"{min_para_hhmm(cb.inicio_evento)}--{min_para_hhmm(cb.fim_evento)}"],
        ["Penalidade (cenário B)", f"{formatar_num(cb.penalidade_fixa, 0)} EUR/restrição + "
                                   f"{formatar_num(cb.penalidade_por_min, 0)} EUR/min"],
    ]
    escrever_tabela_latex(caminho, ["Parâmetro", "Valor"], linhas, "@{}lp{4.6cm}@{}",
                          "Parâmetros do AG da Atividade 1 (implementação própria).",
                          "tab:a1_params")


def tabela_estatisticas(caminho: Path, resumo: dict) -> None:
    chaves = list(resumo["cenarios"])
    S = resumo["cenarios"]

    def linha(rotulo, fn):
        return [rotulo] + [fn(S[k]) for k in chaves]

    linhas = [
        linha("Ótimo exato $F^*$ (enumeração)", lambda s: formatar_num(s["exato"]["fitness"], 0)),
        "midrule",
        linha("Média do melhor $F$", lambda s: formatar_num(s["ag"]["melhor"]["media"], 1)),
        linha("Variância", lambda s: formatar_num(s["ag"]["melhor"]["variancia"], 1)),
        linha("Desvio padrão", lambda s: formatar_num(s["ag"]["melhor"]["desvio_padrao"], 1)),
        linha("Mínimo", lambda s: formatar_num(s["ag"]["melhor"]["minimo"], 0)),
        linha("Mediana", lambda s: formatar_num(s["ag"]["melhor"]["mediana"], 1)),
        linha("Máximo", lambda s: formatar_num(s["ag"]["melhor"]["maximo"], 0)),
        linha("Gap médio para $F^*$", lambda s: formatar_num(s["ag"]["gap_medio_pct"], 2) + "\\%"),
        linha("Execuções com $F=F^*$", lambda s: f"{s['ag']['sucessos']}/{s['ag']['execucoes']}"),
        linha("Geração p/ atingir $F^*$ (média)",
              lambda s: formatar_num(s["ag"]["geracao_acerto"]["media"], 1)
              if s["ag"]["geracao_acerto"] else "--"),
        linha("$\\bar F$ da pop. final (média)",
              lambda s: formatar_num(s["ag"]["media_pop_final"]["media"], 1)),
        linha("$\\sigma_F$ da pop. final (média)",
              lambda s: formatar_num(s["ag"]["desvio_pop_final"]["media"], 1)),
        linha("Tempo por execução (s)", lambda s: formatar_num(s["ag"]["tempo_s"]["media"], 2)),
        "midrule",
        linha("Busca aleatória: média", lambda s: formatar_num(s["aleatoria"]["media"], 1)),
        linha("Busca aleatória: desvio padrão", lambda s: formatar_num(s["aleatoria"]["desvio_padrao"], 1)),
    ]
    escrever_tabela_latex(caminho, [f"Métrica ({resumo['execucoes']} execuções)"]
                          + [f"Cen. {k}" for k in chaves], linhas,
                          "@{}l" + "r" * len(chaves) + "@{}",
                          "Estatísticas do melhor fitness final $F$ (EUR) do AG na Atividade 1.",
                          "tab:a1_stats",
                          notas="Variância e desvio padrão amostrais ($n-1$). A busca aleatória usa "
                                "o mesmo número de avaliações do AG.")


def tabela_roteiro(caminho: Path, sol: dict, cen: Cenario) -> None:
    linhas = []
    for r in sol["linhas"]:
        linhas.append([
            r["origem"],
            f"{min_para_hhmm(r['ida_partida'])}--{min_para_hhmm(r['ida_chegada'])}", f"{r['ida_preco']}",
            f"{r['espera_ida']}",
            f"{min_para_hhmm(r['volta_partida'])}--{min_para_hhmm(r['volta_chegada'])}",
            f"{r['volta_preco']}", f"{r['espera_volta']}",
        ])
    linhas.append("midrule")
    linhas.append(["Total", "", formatar_num(sum(r["ida_preco"] for r in sol["linhas"]), 0),
                   formatar_num(sol["espera_ida"], 0), "",
                   formatar_num(sum(r["volta_preco"] for r in sol["linhas"]), 0),
                   formatar_num(sol["espera_volta"], 0)])
    van = (f"Van sai de FCO às {min_para_hhmm(sol['van_saida'])} e deixa o grupo em FCO às "
           f"{min_para_hhmm(sol['van_entrega'])}. Custo = {formatar_num(sol['custo'], 0)} EUR; "
           f"espera = {formatar_num(sol['espera'], 0)} min; $F$ = {formatar_num(sol['fitness'], 0)}.")
    escrever_tabela_latex(caminho, ["Origem", "Ida", "EUR", "Esp.", "Volta", "EUR", "Esp."],
                          linhas, "@{}lrrrrrr@{}",
                          f"Melhor roteiro encontrado pelo AG -- cenário {cen.chave} ({cen.nome}). "
                          "Esperas em minutos.",
                          f"tab:a1_best_{cen.chave}", notas=van, espaco_colunas="3.5pt")


def tabela_sensibilidade(caminho: Path, dados: list[dict]) -> None:
    linhas, anterior = [], None
    for d in dados:
        if anterior is not None and d["fator"] != anterior:
            linhas.append("midrule")
        valor = d["valor"]
        texto = NOMES_CRUZAMENTO.get(valor, None) or (
            formatar_num(valor, 2) if isinstance(valor, float) else str(valor))
        if d["eh_base"]:
            texto = f"\\textbf{{{texto}}}"
        linhas.append([d["fator"] if d["fator"] != anterior else "", texto,
                       formatar_num(d["media"], 1), formatar_num(d["desvio_padrao"], 1),
                       f"{d['sucessos']}/{d['execucoes']}",
                       formatar_num(d["geracao_acerto"], 0) if d["geracao_acerto"] is not None else "--"])
        anterior = d["fator"]
    escrever_tabela_latex(caminho, ["Fator", "Valor", "Média $F$", "DP", "Sucesso", "Ger."], linhas,
                          "@{}llrrrr@{}",
                          "Análise de sensibilidade (um fator por vez) no cenário A. "
                          "Em negrito, o valor da configuração base.",
                          "tab:a1_sens",
                          notas="Sucesso: execuções que atingiram o ótimo exato. "
                                "Ger.: geração média em que $F^*$ foi atingido.")


def estudo_sensibilidade(tabela: TabelaVoos, cen: Cenario, base: ParametrosAG, execucoes: int,
                         seed: int, f_otimo: float) -> list[dict]:
    cache: dict[ParametrosAG, dict] = {}
    saida = []
    for rotulo, campo, valores in SENSIBILIDADE:
        for v in valores:
            p = replace(base, **{campo: v})
            if p not in cache:
                cache[p] = metricas_lote(executar_lote(tabela, cen, p, execucoes, seed), f_otimo)
            m = cache[p]
            saida.append({
                "fator": rotulo, "campo": campo, "valor": v, "eh_base": p == base,
                "media": m["melhor"]["media"], "desvio_padrao": m["melhor"]["desvio_padrao"],
                "variancia": m["melhor"]["variancia"],
                "sucessos": m["sucessos"], "execucoes": m["execucoes"],
                "geracao_acerto": m["geracao_acerto"]["media"] if m["geracao_acerto"] else None,
            })
            print(f"  [sens] {campo}={v}: media={m['melhor']['media']:.1f} "
                  f"sucesso={m['sucessos']}/{m['execucoes']}", flush=True)
    return saida


def estudo_custo_espera(tabela: TabelaVoos, cen: Cenario, sentidos, base: ParametrosAG,
                        execucoes: int, seed: int, n_amostra: int = 4000) -> dict:
    ida, volta = sentidos
    rng = np.random.default_rng(seed)
    fi = np.flatnonzero(ida.aval.n_violacoes == 0)
    fv = np.flatnonzero(volta.aval.n_violacoes == 0)
    ii, iv = rng.choice(fi, n_amostra), rng.choice(fv, n_amostra)
    amostra = np.column_stack([ida.aval.custo[ii] + volta.aval.custo[iv],
                               ida.aval.espera[ii] + volta.aval.espera[iv]]).astype(float)
    fronteira = fronteira_pareto(tabela, cen, sentidos)
    pontos_ag, pontos_exatos, linhas = [], [], []
    for lam in LAMBDAS:
        cen_l = cen.com_peso(lam)
        ex = otimo_exato(tabela, cen_l, sentidos)
        d = descrever_solucao(tabela, cen_l, ex.individuo)
        pontos_exatos.append((lam, d["custo"], d["espera"]))
        resultados = executar_lote(tabela, cen_l, base, execucoes, seed)
        acertos = 0
        for r in resultados:
            dr = descrever_solucao(tabela, cen_l, r.melhor_individuo)
            pontos_ag.append((lam, dr["custo"], dr["espera"]))
            acertos += r.geracao_que_atingiu(ex.fitness) is not None
        f_ag = np.array([r.melhor_fitness for r in resultados])
        gap = 100.0 * (f_ag - ex.fitness) / ex.fitness
        linhas.append({"cenario": cen.chave, "lambda": lam, "F_exato": ex.fitness,
                       "custo_exato": d["custo"], "espera_exata": d["espera"],
                       "F_medio_ag": f_ag.mean(), "gap_medio_pct": gap.mean(),
                       "gap_max_pct": gap.max(), "sucessos_ag": acertos, "execucoes": execucoes})
    return {"chave": cen.chave, "amostra": amostra, "fronteira": fronteira, "ag": pontos_ag,
            "exato": pontos_exatos, "linhas": linhas}


def principal(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", "--data", type=Path, default=RAIZ / "data" / "flights.txt")
    ap.add_argument("--saida", "--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--execucoes", "--runs", type=int, default=30)
    ap.add_argument("--rapido", "--quick", action="store_true")
    ap.add_argument("--sem-sensibilidade", "--skip-sensitivity", action="store_true")
    ap.add_argument("--sem-custo-espera", "--skip-tradeoff", action="store_true")
    args = ap.parse_args(argv)

    if args.saida is None:
        args.saida = RAIZ / "results" / ("quick" if args.rapido else "") / "atividade1"
    estilo_graficos.aplicar_estilo()
    saida, pasta_fig, pasta_tab = args.saida, args.saida / "figures", args.saida / "tables"
    for d in (saida, pasta_fig, pasta_tab):
        d.mkdir(parents=True, exist_ok=True)

    base = PARAMETROS_BASE
    execucoes = args.execucoes
    exec_sens, exec_custo = execucoes, 10
    if args.rapido:
        base = replace(base, geracoes=60)
        execucoes, exec_sens, exec_custo = 5, 3, 2

    tabela = carregar_voos(args.dados)
    print(f"Espaco de busca: {tabela.tamanho_espaco:.3e} cromossomos | genes: {tabela.nomes_genes()}")
    print(f"Parametros: {base.como_dict()}")

    resumo = {"parametros": base.como_dict(), "seed": args.seed, "execucoes": execucoes, "cenarios": {}}
    todos, otimo, melhores_ag, melhores_aleat, sentidos_cache = {}, {}, {}, {}, {}

    for chave, cen in CENARIOS.items():
        print(f"\n== {cen.descrever()}")
        t0 = time.perf_counter()
        sentidos = enumerar_sentidos(tabela, cen)
        sentidos_cache[chave] = sentidos
        ex = otimo_exato(tabela, cen, sentidos)
        t_enum = time.perf_counter() - t0
        print(f"Otimo exato F* = {ex.fitness:.1f} ({ex.n_otimos} cromossomo(s); "
              f"enumeracao de 2x10^6 combinacoes em {t_enum:.2f} s)")

        resultados = executar_lote(tabela, cen, base, execucoes, args.seed)
        metricas = metricas_lote(resultados, ex.fitness)
        fitness = criar_fitness(tabela, cen)
        aleat = [busca_aleatoria(fitness, tabela.limite_genes, base.avaliacoes,
                                 seed=args.seed + 10_000 + r)[1]
                 for r in range(execucoes)]

        melhor_exec = min(resultados, key=lambda r: r.melhor_fitness)
        sol = descrever_solucao(tabela, cen, melhor_exec.melhor_individuo)
        texto = formatar_solucao(sol)
        print(texto)
        print(f"AG: media={metricas['melhor']['media']:.1f} var={metricas['melhor']['variancia']:.1f} "
              f"dp={metricas['melhor']['desvio_padrao']:.1f} sucesso={metricas['sucessos']}/{execucoes} | "
              f"busca aleatoria: media={np.mean(aleat):.1f}")

        (saida / f"melhor_solucao_{chave}.txt").write_text(cen.descrever() + "\n" + texto + "\n",
                                                           encoding="utf-8")
        salvar_csv(saida / f"execucoes_{chave}.csv", [{
            "execucao": i, "seed": r.seed, "melhor_fitness": r.melhor_fitness,
            "geracao_otimo": r.geracao_que_atingiu(ex.fitness),
            "media_pop_final": r.fitness_final.mean(),
            "desvio_pop_final": r.fitness_final.std(ddof=1), "tempo_s": r.tempo_s,
            "genes": " ".join(map(str, r.melhor_individuo)), "melhor_busca_aleatoria": aleat[i],
        } for i, r in enumerate(resultados)])
        np.savez_compressed(saida / f"historicos_{chave}.npz",
                            **{m: np.vstack([r.historico[m] for r in resultados])
                               for m in ("melhor", "medio", "pior", "desvio", "unicos")})
        tabela_roteiro(pasta_tab / f"tab_a1_roteiro_{chave}.tex", sol, cen)
        graficos.grafico_roteiro(sol, cen, pasta_fig, f"a1_roteiro_{chave}")

        resumo["cenarios"][chave] = {
            "descricao": cen.descrever(),
            "exato": {"fitness": ex.fitness, "genes": ex.individuo, "n_otimos": ex.n_otimos,
                      "tempo_enumeracao_s": t_enum},
            "ag": metricas,
            "aleatoria": resumir(aleat),
            "melhor_solucao": sol,
        }
        todos[chave], otimo[chave] = resultados, ex.fitness
        melhores_ag[chave] = [r.melhor_fitness for r in resultados]
        melhores_aleat[chave] = aleat

    graficos.grafico_convergencia(todos, otimo, CENARIOS, pasta_fig, "a1_convergencia")
    graficos.grafico_distribuicao_final(melhores_ag, melhores_aleat, otimo, pasta_fig,
                                        "a1_distribuicao_final")
    tabela_parametros(pasta_tab / "tab_a1_parametros.tex", base, execucoes, args.seed)
    tabela_estatisticas(pasta_tab / "tab_a1_estatisticas.tex", resumo)

    if not args.sem_sensibilidade:
        print("\n== Analise de sensibilidade (cenario A)")
        sens = estudo_sensibilidade(tabela, CENARIOS["A"], base, exec_sens, args.seed, otimo["A"])
        salvar_csv(saida / "sensibilidade_A.csv", sens)
        tabela_sensibilidade(pasta_tab / "tab_a1_sensibilidade.tex", sens)
        resumo["sensibilidade"] = sens

    if not args.sem_custo_espera:
        print("\n== Varredura de lambda (custo x espera)")
        paineis = [estudo_custo_espera(tabela, cen, sentidos_cache[k], base, exec_custo, args.seed)
                   for k, cen in CENARIOS.items()]
        graficos.grafico_custo_espera(paineis, pasta_fig, "a1_custo_vs_espera")
        linhas = [r for p in paineis for r in p["linhas"]]
        salvar_csv(saida / "custo_espera_lambda.csv", linhas)
        resumo["custo_espera"] = linhas
        resumo["fronteira_pareto"] = {p["chave"]: p["fronteira"] for p in paineis}
        for r in linhas:
            print(f"  {r['cenario']} lambda={r['lambda']}: C={r['custo_exato']:.0f} "
                  f"W={r['espera_exata']:.0f} sucesso AG={r['sucessos_ag']}/{r['execucoes']} "
                  f"gap medio={r['gap_medio_pct']:.2f}% gap max={r['gap_max_pct']:.2f}%")

    salvar_json(saida / "resumo.json", resumo)
    print(f"\nResultados salvos em {saida}")
    return resumo


if __name__ == "__main__":
    principal()

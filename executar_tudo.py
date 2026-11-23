"""Uso (a partir da raiz do projeto):

    python executar_tudo.py
    python executar_tudo.py --rapido
    python executar_tudo.py --seed 7 --execucoes 50 --processos 4
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

from src.atividade1_voos import experimentos as exp1
from src.atividade2_rastrigin import experimentos as exp2

RAIZ = Path(__file__).resolve().parent


def exportar_para_artigo(pastas_resultados: list[Path], pasta_artigo: Path) -> int:
    n = 0
    for pasta in pastas_resultados:
        for sub, padrao in (("figures", "*.pdf"), ("tables", "*.tex")):
            destino = pasta_artigo / sub
            destino.mkdir(parents=True, exist_ok=True)
            for arq in sorted((pasta / sub).glob(padrao)):
                shutil.copy2(arq, destino / arq.name)
                n += 1
    return n


def principal() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--execucoes", "--runs", type=int, default=30)
    ap.add_argument("--processos", "--jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--rapido", "--quick", action="store_true")
    ap.add_argument("--sem-exportar", "--no-export", action="store_true")
    args = ap.parse_args()

    comuns = ["--seed", str(args.seed), "--execucoes", str(args.execucoes)]
    if args.rapido:
        comuns.append("--rapido")
    t0 = time.perf_counter()
    print("#" * 70 + "\n# Atividade 1 - Voos (AG implementado do zero)\n" + "#" * 70)
    exp1.principal(comuns)
    print("\n" + "#" * 70 + "\n# Atividade 2 - Rastrigin (DEAP)\n" + "#" * 70)
    exp2.principal(comuns + ["--processos", str(args.processos)])

    if not args.sem_exportar and not args.rapido:
        n = exportar_para_artigo([RAIZ / "results" / "atividade1", RAIZ / "results" / "atividade2"],
                                 RAIZ / "paper")
        print(f"\n{n} arquivos copiados para paper/figures e paper/tables")
    print(f"Tempo total: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    principal()

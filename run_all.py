"""Executa todos os experimentos e atualiza figuras/tabelas usadas pelo artigo.

Uso::

    python run_all.py                 # completo (~5-10 min em 4 núcleos)
    python run_all.py --quick         # versão reduzida, só para testar o pipeline
    python run_all.py --seed 7 --jobs 4

Saídas:
    results/atividade1/  e  results/atividade2/   (CSV, JSON, NPZ, figuras, tabelas)
    paper/figures/  e  paper/tables/              (cópias usadas pelo LaTeX)
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

from src.atividade1_voos import experiments as exp1
from src.atividade2_rastrigin import experiments as exp2

ROOT = Path(__file__).resolve().parent


def export_to_paper(results_dirs: list[Path], paper_dir: Path) -> int:
    """Copia as figuras (PDF) e tabelas (.tex) geradas para a pasta do artigo."""
    n = 0
    for d in results_dirs:
        for sub, pattern in (("figures", "*.pdf"), ("tables", "*.tex")):
            dst = paper_dir / sub
            dst.mkdir(parents=True, exist_ok=True)
            for f in sorted((d / sub).glob(pattern)):
                shutil.copy2(f, dst / f.name)
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-export", action="store_true", help="não copia resultados para paper/")
    args = ap.parse_args()

    common = ["--seed", str(args.seed), "--runs", str(args.runs)] + (["--quick"] if args.quick else [])
    t0 = time.perf_counter()
    print("#" * 70 + "\n# Atividade 1 - Voos (AG implementado do zero)\n" + "#" * 70)
    exp1.main(common)
    print("\n" + "#" * 70 + "\n# Atividade 2 - Rastrigin (DEAP)\n" + "#" * 70)
    exp2.main(common + ["--jobs", str(args.jobs)])

    if not args.no_export and not args.quick:
        n = export_to_paper([ROOT / "results" / "atividade1", ROOT / "results" / "atividade2"],
                            ROOT / "paper")
        print(f"\n{n} arquivos copiados para paper/figures e paper/tables")
    print(f"Tempo total: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()

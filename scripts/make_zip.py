"""Gera o arquivo .zip de entrega (SIGAA) com código, dados, resultados e artigo.

Uso::

    python scripts/make_zip.py                       # -> dist/ECOI2217_N2_AG.zip
    python scripts/make_zip.py --name MeuNome_N2.zip

Ficam de fora: .git, caches do Python/pytest, arquivos auxiliares do LaTeX,
results/quick/ (saídas do modo de teste) e a própria pasta dist/.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", "dist", ".venv", "venv", ".idea", ".vscode"}
EXCLUDE_SUFFIXES = {".pyc", ".aux", ".log", ".out", ".blg", ".fls", ".fdb_latexmk", ".synctex.gz",
                    ".toc", ".bbl-SAVE-ERROR"}


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts) or rel.parts[:2] == ("results", "quick"):
            continue
        if path.is_file() and not any(str(path).endswith(s) for s in EXCLUDE_SUFFIXES):
            yield path, rel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="ECOI2217_N2_AG.zip")
    args = ap.parse_args()
    out = ROOT / "dist" / args.name
    out.parent.mkdir(exist_ok=True)
    top = Path(args.name).stem
    n = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, rel in iter_files(ROOT):
            zf.write(path, Path(top) / rel)
            n += 1
    print(f"{out} ({n} arquivos, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

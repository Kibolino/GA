"""Uso (a partir da raiz do projeto):

    python scripts/gerar_zip.py
    python scripts/gerar_zip.py --nome MeuNome_N2.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PASTAS_EXCLUIDAS = {".git", "__pycache__", ".pytest_cache", "dist", ".venv", "venv", ".idea", ".vscode"}
SUFIXOS_EXCLUIDOS = {".pyc", ".aux", ".log", ".out", ".blg", ".fls", ".fdb_latexmk", ".synctex.gz",
                     ".toc", ".bbl-SAVE-ERROR"}


def listar_arquivos(raiz: Path):
    for caminho in sorted(raiz.rglob("*")):
        relativo = caminho.relative_to(raiz)
        if (any(parte in PASTAS_EXCLUIDAS for parte in relativo.parts)
                or relativo.parts[:2] == ("results", "quick")):
            continue
        if caminho.is_file() and not any(str(caminho).endswith(s) for s in SUFIXOS_EXCLUIDOS):
            yield caminho, relativo


def principal() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nome", "--name", default="ECOI2217_N2_AG.zip")
    args = ap.parse_args()
    destino = RAIZ / "dist" / args.nome
    destino.parent.mkdir(exist_ok=True)
    topo = Path(args.nome).stem
    n = 0
    with zipfile.ZipFile(destino, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for caminho, relativo in listar_arquivos(RAIZ):
            zf.write(caminho, Path(topo) / relativo)
            n += 1
    print(f"{destino} ({n} arquivos, {destino.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    principal()

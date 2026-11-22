"""Estatística descritiva e utilitários de exportação de resultados.

Todas as variâncias e desvios padrão são *amostrais* (ddof=1), pois as
execuções independentes do AG são tratadas como uma amostra da distribuição
de resultados do algoritmo.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def summarize(values: Iterable[float]) -> dict[str, float]:
    """Resumo estatístico de uma amostra (média, variância, desvio padrão etc.)."""
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        raise ValueError("amostra vazia")
    ddof = 1 if x.size > 1 else 0
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "var": float(x.var(ddof=ddof)),
        "std": float(x.std(ddof=ddof)),
        "min": float(x.min()),
        "median": float(np.median(x)),
        "max": float(x.max()),
    }


def history_band(histories: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Média e desvio padrão, geração a geração, de várias curvas de convergência."""
    h = np.vstack(histories)
    ddof = 1 if h.shape[0] > 1 else 0
    return h.mean(axis=0), h.std(axis=0, ddof=ddof)


def to_builtin(obj):
    """Converte recursivamente tipos numpy para tipos nativos (serialização JSON)."""
    if isinstance(obj, Mapping):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_builtin(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(data), indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(to_builtin(dict(row)))


def fmt_num(value: float, decimals: int = 2) -> str:
    """Formata número no padrão brasileiro para as tabelas LaTeX (vírgula decimal).

    Usa ``{,}`` para que o LaTeX não insira espaço após a vírgula em modo matemático.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    text = f"{value:,.{decimals}f}"
    # 1,234.56 -> 1.234,56
    text = text.replace(",", "\0").replace(".", ",").replace("\0", ".")
    return text.replace(",", "{,}")


def fmt_sci(value: float, decimals: int = 2) -> str:
    """Notação científica em LaTeX, p.ex. 1{,}23\\times10^{-4}."""
    if value == 0:
        return "0"
    mant, exp = f"{value:.{decimals}e}".split("e")
    mant = mant.replace(".", "{,}")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def write_latex_table(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]],
                      colspec: str, caption: str, label: str, wide: bool = False,
                      size: str = "\\footnotesize", notes: str | None = None,
                      tabcolsep: str = "4pt") -> None:
    """Gera um ambiente ``table`` (booktabs) pronto para ``\\input`` no artigo."""
    env = "table*" if wide else "table"
    lines = [
        f"% Arquivo gerado automaticamente por src/ -- nao editar manualmente.",
        f"\\begin{{{env}}}[!t]",
        "\\centering",
        size,
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row == "midrule":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines.append(f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\scriptsize {notes}}}")
    lines.append(f"\\end{{{env}}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

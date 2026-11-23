from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def resumir(valores: Iterable[float]) -> dict[str, float]:
    x = np.asarray(list(valores), dtype=float)
    if x.size == 0:
        raise ValueError("amostra vazia")
    ddof = 1 if x.size > 1 else 0
    return {
        "n": int(x.size),
        "media": float(x.mean()),
        "variancia": float(x.var(ddof=ddof)),
        "desvio_padrao": float(x.std(ddof=ddof)),
        "minimo": float(x.min()),
        "mediana": float(np.median(x)),
        "maximo": float(x.max()),
    }


def faixa_historico(historicos: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    h = np.vstack(historicos)
    ddof = 1 if h.shape[0] > 1 else 0
    return h.mean(axis=0), h.std(axis=0, ddof=ddof)


def para_tipos_nativos(obj):
    if isinstance(obj, Mapping):
        return {str(k): para_tipos_nativos(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [para_tipos_nativos(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return para_tipos_nativos(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def salvar_json(caminho: Path, dados) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(para_tipos_nativos(dados), indent=2, ensure_ascii=False),
                       encoding="utf-8")


def salvar_csv(caminho: Path, linhas: Sequence[Mapping], campos: Sequence[str] | None = None) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if not linhas:
        caminho.write_text("", encoding="utf-8")
        return
    campos = list(campos or linhas[0].keys())
    with caminho.open("w", newline="", encoding="utf-8") as arq:
        escritor = csv.DictWriter(arq, fieldnames=campos)
        escritor.writeheader()
        for linha in linhas:
            escritor.writerow(para_tipos_nativos(dict(linha)))


def formatar_num(valor: float, casas: int = 2) -> str:
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return "--"
    texto = f"{valor:,.{casas}f}"
    texto = texto.replace(",", "\0").replace(".", ",").replace("\0", ".")
    return texto.replace(",", "{,}")


def formatar_cientifico(valor: float, casas: int = 2) -> str:
    if valor == 0:
        return "0"
    mantissa, expoente = f"{valor:.{casas}e}".split("e")
    mantissa = mantissa.replace(".", "{,}")
    return f"${mantissa}\\times10^{{{int(expoente)}}}$"


def escrever_tabela_latex(caminho: Path, cabecalho: Sequence[str], linhas: Sequence[Sequence[str]],
                          colunas: str, legenda: str, rotulo: str, largura_total: bool = False,
                          tamanho: str = "\\footnotesize", notas: str | None = None,
                          espaco_colunas: str = "4pt") -> None:
    ambiente = "table*" if largura_total else "table"
    texto = [
        f"\\begin{{{ambiente}}}[!t]",
        "\\centering",
        tamanho,
        f"\\setlength{{\\tabcolsep}}{{{espaco_colunas}}}",
        f"\\caption{{{legenda}}}",
        f"\\label{{{rotulo}}}",
        f"\\begin{{tabular}}{{{colunas}}}",
        "\\toprule",
        " & ".join(cabecalho) + " \\\\",
        "\\midrule",
    ]
    for linha in linhas:
        if linha == "midrule":
            texto.append("\\midrule")
        else:
            texto.append(" & ".join(linha) + " \\\\")
    texto += ["\\bottomrule", "\\end{tabular}"]
    if notas:
        texto.append(f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\scriptsize {notas}}}")
    texto.append(f"\\end{{{ambiente}}}")
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text("\n".join(texto) + "\n", encoding="utf-8")

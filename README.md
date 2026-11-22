# ECOI2217 – Inteligência Artificial · Atividade N2: Algoritmos Genéticos

Duas aplicações de Algoritmos Genéticos (AG):

| Atividade | Problema | Implementação |
|---|---|---|
| **1** | Passagens de 6 pesquisadores (LIS, MAD, CDG, DUB, BRU, LHR) para um congresso em Roma (FCO): minimizar o **preço** e o **tempo de espera** no aeroporto (`data/flights.txt`) | **Do zero**: só `numpy` e `matplotlib` |
| **2** | Minimizar a função de **Rastrigin** com A = 10, x<sub>i</sub> ∈ [−5,12; 5,12], para **n = 2** e **n = 10** | Biblioteca **DEAP** |

O artigo, com modelagem, experimentos, estatísticas e um tutorial de DEAP, está em [`paper/main.tex`](paper/main.tex). A versão compilada é [`paper/main.pdf`](paper/main.pdf).

---

## 1. Instalação

Requer Python ≥ 3.10.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

## 2. Como executar

Execute tudo a partir da raiz do projeto:

```bash
python run_all.py
```

Esse comando roda as duas atividades: 30 execuções por configuração, a análise de sensibilidade e a varredura de λ. Ele grava os resultados em `results/` e copia figuras e tabelas para `paper/`. Leva cerca de 5–10 min em 4 núcleos.

Opções úteis:

```bash
python run_all.py --quick                 # versão reduzida (~20 s), grava em results/quick/
```

```bash
python run_all.py --seed 7 --runs 50 --jobs 4
```

Cada atividade também pode ser executada separadamente:

```bash
python -m src.atividade1_voos.experiments --seed 2026 --runs 30
```

```bash
python -m src.atividade2_rastrigin.experiments --seed 2026 --runs 30 --jobs 4 --dims 2 10
```

Testes automatizados: validade dos operadores, fitness calculada à mão, reprodutibilidade e mínimo da Rastrigin.

```bash
python -m pytest
```

Compilar o artigo (ou enviar a pasta `paper/` ao Overleaf):

```bash
cd paper && latexmk -pdf main.tex
```

Gerar o `.zip` de entrega para o SIGAA (sai em `dist/`):

```bash
python scripts/make_zip.py
```

### Reprodutibilidade

A execução *r* usa a seed `seed + r` (padrão `seed = 2026`). A Atividade 1 usa `numpy.random.default_rng(seed)`. A Atividade 2 usa `random.seed(seed)`, que é o gerador usado internamente pelo DEAP. Com a mesma seed, os resultados são idênticos, inclusive com `--jobs > 1`.

---

## 3. Organização

```
.
├── data/flights.txt                 # espaço de busca da Atividade 1 (120 voos)
├── docs/                            # enunciado (PDF) e template IEEE fornecido
├── src/
│   ├── common/                      # estatística (média, variância, DP), estilo de gráficos, tabelas LaTeX
│   ├── atividade1_voos/             # AG implementado do zero
│   │   ├── flights.py               #   leitura e organização dos voos
│   │   ├── model.py                 #   cenários (premissas), decodificação e função fitness
│   │   ├── ga.py                    #   motor do AG: torneio, cruzamento, mutações, elitismo
│   │   ├── exact.py                 #   ótimo exato por enumeração decomposta (só validação)
│   │   ├── plots.py                 #   gráficos
│   │   └── experiments.py           #   experimentos (CLI)
│   └── atividade2_rastrigin/        # AG com DEAP
│       ├── rastrigin.py             #   função objetivo
│       ├── ga_deap.py               #   creator/toolbox, laço elitista, busca aleatória
│       ├── plots.py
│       └── experiments.py
├── tests/                           # pytest
├── results/atividade1|2/            # CSV por execução, resumo.json, históricos .npz, figuras, tabelas
├── paper/                           # main.tex, referencias.bib, figures/, tables/ (gerados), main.pdf
├── run_all.py                       # pipeline completo
└── scripts/make_zip.py              # empacotamento para o SIGAA
```

---

## 4. Atividade 1 – modelagem resumida

**Cromossomo:** vetor de 12 inteiros `[ida_LIS, volta_LIS, ida_MAD, …, ida_LHR, volta_LHR]`. Cada gene é o índice (0–9) de um voo real da rota correspondente, com os voos ordenados por horário de partida. Qualquer cromossomo representa um conjunto de voos existente, então cruzamento e mutação **não conseguem criar voos novos**. O espaço de busca tem 10<sup>12</sup> soluções.

**Fitness (minimização):** `F = C + λ·W + P`

- `C`: soma dos preços das 12 passagens (EUR).
- `W`: espera total no aeroporto de FCO, em minutos, somada sobre os 6 pesquisadores.
- `λ = 1 EUR/min`: converte tempo em custo; é configurável.
- `P`: penalidade por violar a janela do congresso (só no cenário B).

**Premissas:**

1. Uma única van leva o grupo ao congresso e só sai de FCO quando o **último** pesquisador chega. A espera na ida é `última_chegada − chegada_i`.
2. Na volta, a van deixa o grupo junto em FCO:
   - **Cenário A** (ida e volta em dias distintos): a van chega 60 min (check-in) antes do primeiro voo de volta.
   - **Cenário B** (mesa-redonda das 14:00 às 16:00 no mesmo dia): a van sai do congresso às 16:00 e chega a FCO às 16:45, após 45 min de translado.
3. Restrições do cenário B:
   - `última_chegada + 45 min ≤ 14:00`;
   - `partida_i ≥ 16:45 + 60 min` de check-in.
   - Cada violação custa 1000 EUR, mais 10 EUR por minuto de violação. A penalidade proporcional guia o AG de volta à região factível.

**Operadores:**

- seleção por torneio (k = 3);
- cruzamento uniforme (p<sub>c</sub> = 0,9);
- mutação por gene (p<sub>m</sub> = 0,05), metade *creep* ±1 (voo vizinho no horário) e metade *reset* (outro voo da rota);
- **mutação de bloco** (p<sub>shift</sub> = 0,3): desloca todos os voos de ida, ou todos os de volta, para o voo vizinho, preservando a sincronia do grupo;
- elitismo dos 5 melhores.

População de 150 indivíduos, 400 gerações.

**Validação:** a fitness se separa em uma parte de ida e outra de volta, `F = F_ida + F_volta`. Por isso o ótimo global exato sai por enumeração de 2 × 10<sup>6</sup> combinações (< 1 s). Esse valor é usado apenas para medir a taxa de acerto do AG.

## 5. Atividade 2 – modelagem resumida

**Cromossomo:** lista de *n* reais em [−5,12; 5,12] (`creator.Individual`).

**Operadores DEAP limitados ao domínio:**

- `cxSimulatedBinaryBounded` (SBX, η<sub>c</sub> = 15, p<sub>c</sub> = 0,9);
- `mutPolynomialBounded` (η<sub>m</sub> = 20, probabilidade 1/n por gene);
- `selTournament` (k = 3).

**Elitismo:** laço geracional próprio (`ea_elitist`) que reinsere os 2 melhores. O `eaSimple` do DEAP não reinsere os melhores; o `HallOfFame` apenas os guarda.

**Configurações:** n = 2 com N = 100 e G = 100; n = 10 com N = 200 e G = 500.

## 6. Resultados

O resumo numérico de cada atividade está em `results/atividade*/resumo.json`. As tabelas e figuras correspondentes ficam em `results/atividade*/{tables,figures}/` e são discutidas no artigo.

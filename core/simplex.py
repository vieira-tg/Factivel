"""Método Simplex (Duas Fases) para N variáveis, com passos para animação.

Resolve um PL com qualquer número de variáveis usando o Simplex clássico na
variante **Duas Fases**:

1. **Normalização**: garante `b >= 0` (multiplicando a restrição por -1),
   converte `<=` em variável de folga (+s), `>=` em excesso (-s) + artificial
   (+a) e `=` em artificial (+a). Assume **não-negatividade** das variáveis
   (x_j >= 0), que é o pressuposto padrão do Simplex — a razão mínima e a base
   inicial mantêm x >= 0 automaticamente. Minimizar vira maximizar com objetivo
   negado (e o sinal de Z é revertido no final).
2. **Fase 1**: minimiza a soma das artificiais (equivalente a maximizar -soma).
   Se no ótimo a soma ainda for > 0, o problema é inviável. Senão, remove as
   colunas artificiais e segue.
3. **Fase 2**: maximiza o objetivo original sobre a base viável encontrada.

Cada pivoteamento gera um novo tableau. `resolver_simplex` devolve, além do
resultado, uma LISTA DE PASSOS prontos para exibição animada: cada pivô vira
passos que destacam progressivamente a coluna pivô, a linha pivô e o elemento
pivô, seguidos do tableau recalculado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .modelo import STATUS_ILIMITADO, STATUS_INVIAVEL, STATUS_OTIMO

# Tolerância numérica (nunca comparar float com ==).
EPS = 1e-9


@dataclass
class PassoSimplex:
    """Um estado do tableau pronto para renderização no QTableWidget.

    - `matriz`: matriz completa — linhas de restrição + uma linha Z, colunas de
      variáveis + a última coluna do lado direito (RHS).
    - `colunas`: rótulos das colunas de variáveis (ex.: ["x1","x2","s1"...]).
    - `rotulos_linhas`: variável básica de cada linha de restrição.
    """

    matriz: np.ndarray
    colunas: list[str]
    rotulos_linhas: list[str]
    fase: int
    iteracao: int
    texto: str
    solucao: dict  # {"x": {var: valor}, "z": valor, "completo": {...}}
    destaque_coluna: int | None = None
    destaque_linha: int | None = None
    destaque_pivo: tuple[int, int] | None = None
    otimo: bool = False
    final: bool = False
    status: str = STATUS_OTIMO


@dataclass
class ResultadoSimplex:
    """Resultado completo do método Simplex."""

    status: str = STATUS_OTIMO  # "otimo" | "inviavel" | "ilimitado"
    z_otimo: float | None = None
    solucao: dict | None = None  # {"x1": ..., "x2": ...}
    mensagem: str = ""
    passos: list[PassoSimplex] = field(default_factory=list)


# ------------------------------------------------------------------ utilitários
def _reducidos(a: np.ndarray, b: np.ndarray, c: np.ndarray, base: list[int]):
    """Custos reduzidos e valor corrente do objetivo (maximizar).

    Custo reduzido da coluna j: `c_j - sum_i c_{B_i} * a[i, j]`. Quando todos são
    >= 0, a base é ótima. Devolve `(r, z)`.
    """
    r = np.array(c, dtype=float).copy()
    z = 0.0
    for i, bi in enumerate(base):
        r -= float(c[bi]) * a[i]
        z += float(c[bi]) * b[i]
    return r, z


def _solucao_atual(b: np.ndarray, base: list[int], colunas: list[str], z: float) -> dict:
    """Lê a solução básica corrente: variável básica = RHS, não-básica = 0."""
    completo: dict[str, float] = {}
    for j in range(len(colunas)):
        completo[colunas[j]] = 0.0
    for i, bi in enumerate(base):
        completo[colunas[bi]] = float(b[i])
    originais = {k: v for k, v in completo.items() if k.startswith("x")}
    return {"x": originais, "z": z, "completo": completo}


def _pivot(a: np.ndarray, b: np.ndarray, base: list[int], linha: int, coluna: int):
    """Eliminação de Gauss-Jordan: normaliza a linha pivô e zera a coluna pivô."""
    a = a.copy()
    b = b.copy()
    base = list(base)
    pivo = a[linha, coluna]
    a[linha] = a[linha] / pivo
    b[linha] = b[linha] / pivo
    for i in range(len(a)):
        if i != linha and abs(a[i, coluna]) > EPS:
            fator = a[i, coluna]
            a[i] = a[i] - fator * a[linha]
            b[i] = b[i] - fator * b[linha]
    base[linha] = coluna
    return a, b, base


# ------------------------------------------------------------------ execução de uma fase
def _executar_fase(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    base: list[int],
    colunas: list[str],
    numero_fase: int,
    registros: list[dict],
) -> tuple[list[dict], np.ndarray, np.ndarray, list[int], str]:
    """Executa uma fase, gravando registros de iteração (com custos reduzidos).

    Devolve `(registros, a, b, base, status)`. Cada registro guarda o tableau no
    início da iteração (antes do pivô), a coluna que entra, a linha que sai, o
    pivô, os custos reduzidos `r` e o valor corrente `z`.
    """
    num_iter = 0
    m = len(a)

    while True:
        r, z = _reducidos(a, b, c, base)
        solucao = _solucao_atual(b, base, colunas, z)

        # ---- Critério de parada: nenhum custo reduzido positivo. --------------
        # Em maximização, a variável que entra é a de MAIOR custo reduzido
        # positivo (r_j > 0); quando todos são <= 0, a base é ótima.
        positivos = [j for j in range(len(c)) if r[j] > EPS]
        if not positivos:
            registros.append(
                {
                    "a": a.copy(), "b": b.copy(), "colunas": list(colunas),
                    "base": list(base), "solucao": solucao, "r": r, "z": z,
                    "col": None, "linha": None, "pivo": None,
                    "fase": numero_fase, "iter": num_iter, "status": STATUS_OTIMO,
                    "texto": (
                        f"Fase {numero_fase}: tabela ótima "
                        "(todos os custos reduzidos são ≤ 0)."
                    ),
                }
            )
            return registros, a, b, base, STATUS_OTIMO

        # ---- Coluna pivô: a variável que entra (maior custo reduzido). --------
        coluna = max(positivos, key=lambda j: r[j])

        # ---- Teste da razão mínima: a linha que sai. --------------------------
        melhor: tuple[float, int] | None = None
        for i in range(m):
            if a[i, coluna] > EPS:
                razao = b[i] / a[i, coluna]
                if melhor is None or razao < melhor[0] - EPS:
                    melhor = (razao, i)
                elif abs(razao - melhor[0]) < EPS and i < melhor[1]:
                    melhor = (razao, i)

        if melhor is None:
            # Nenhuma entrada positiva na coluna pivô: direção ilimitada.
            registros.append(
                {
                    "a": a.copy(), "b": b.copy(), "colunas": list(colunas),
                    "base": list(base), "solucao": solucao, "r": r, "z": z,
                    "col": coluna, "linha": None, "pivo": None,
                    "fase": numero_fase, "iter": num_iter, "status": STATUS_ILIMITADO,
                    "texto": (
                        f"Fase {numero_fase}: a coluna {colunas[coluna]} não tem "
                        "entradas positivas → a razão mínima não existe e o "
                        "problema é ILIMITADO."
                    ),
                }
            )
            return registros, a, b, base, STATUS_ILIMITADO

        linha = melhor[1]
        pivo = float(a[linha, coluna])

        registros.append(
            {
                "a": a.copy(), "b": b.copy(), "colunas": list(colunas),
                "base": list(base), "solucao": solucao, "r": r, "z": z,
                "col": coluna, "linha": linha, "pivo": pivo,
                "fase": numero_fase, "iter": num_iter, "status": STATUS_OTIMO,
            }
        )

        # ---- Executa o pivoteamento (gera o tableau da próxima iteração). -----
        a, b, base = _pivot(a, b, base, linha, coluna)
        num_iter += 1


# ------------------------------------------------------------------ Fase 1 → Fase 2
def _remover_colunas_artificiais(a, b, c, colunas, base):
    """Remove as colunas artificiais (rótulo com prefixo 'a')."""
    manter = [j for j, lab in enumerate(colunas) if not lab.startswith("a")]
    novas_colunas = [colunas[j] for j in manter]
    novo_a = a[:, manter]
    novo_c = np.array([c[j] for j in manter], dtype=float)
    novo_b = b.copy()

    mapa = {velho: novo for novo, velho in enumerate(manter)}
    nova_base: list[int] = []
    for bi in base:
        nova_base.append(mapa[bi] if bi in mapa else -1)
    return novo_a, novo_b, novo_c, novas_colunas, nova_base


def _preparar_fase2(a, b, c, colunas, base):
    """Expulsa artificiais da base e remove suas colunas para começar a Fase 2."""
    a = a.copy()
    b = b.copy()
    base = list(base)

    # Se alguma artificial ficou na base (degenerada, valor 0), pivota numa
    # coluna não-artificial para tirá-la.
    for i in range(len(base)):
        if colunas[base[i]].startswith("a"):
            escolhida = None
            for j, lab in enumerate(colunas):
                if not lab.startswith("a") and abs(a[i, j]) > EPS:
                    escolhida = j
                    break
            if escolhida is not None:
                a, b, base = _pivot(a, b, base, i, escolhida)

    a, b, c, colunas, base = _remover_colunas_artificiais(a, b, c, colunas, base)

    # Descarta linhas redundantes (base marcada -1 ⇒ linha 0 = 0).
    linhas_validas = [i for i in range(len(base)) if base[i] != -1]
    if len(linhas_validas) < len(base):
        a = a[linhas_validas, :]
        b = b[linhas_validas]
        base = [base[i] for i in linhas_validas]

    return a, b, c, colunas, base


# ------------------------------------------------------------------ montagem de passos
def _passo_com_destaque(reg: dict, extras: dict) -> dict:
    """Copia um registro e calcula o texto/destaques apropriados."""
    novo = dict(reg)
    novo.update(extras)

    colunas = reg["colunas"]
    col = reg.get("col")
    linha = reg.get("linha")
    fase = reg["fase"]

    if extras.get("inicial"):
        novo["texto"] = f"Fase {fase}: tableau inicial."
    elif extras.get("novo"):
        novo["texto"] = (
            f"Fase {fase}, iteração {reg['iter']}: tableau recalculado "
            "(eliminação de Gauss)."
        )
    elif extras.get("destaque_coluna") is not None and extras.get("destaque_linha") is None:
        novo["texto"] = (
            f"Fase {fase}, iteração {reg['iter']}: a variável {colunas[col]} ENTRA "
            "na base (coluna pivô — maior custo reduzido)."
        )
    elif extras.get("destaque_linha") is not None and extras.get("destaque_pivo") is None:
        sai = colunas[reg["base"][linha]]
        novo["texto"] = (
            f"Fase {fase}, iteração {reg['iter']}: a variável {sai} SAI da base "
            "(linha pivô — menor razão no teste da razão mínima)."
        )
    elif extras.get("destaque_pivo") is not None:
        sai = colunas[reg["base"][linha]]
        novo["texto"] = (
            f"Fase {fase}, iteração {reg['iter']}: elemento pivô = {reg['pivo']:.4g} "
            f"(entra {colunas[col]}, sai {sai})."
        )
    elif extras.get("final_destaque"):
        novo["otimo"] = True
        novo["final"] = True
        # Mantém o texto de ótimo/ilimitado já gravado no registro terminal.
    return novo


def _formatar_passo(reg: dict) -> PassoSimplex:
    """Constrói um PassoSimplex (matriz completa + rótulos) a partir de um registro."""
    a = np.array(reg["a"], dtype=float)
    b = np.array(reg["b"], dtype=float)
    colunas = list(reg["colunas"])
    base = list(reg["base"])
    r = np.array(reg["r"], dtype=float)
    z = float(reg["z"])

    # Matriz de exibição: linhas de restrição (coeficientes + RHS) e, por fim, a
    # linha Z com os custos reduzidos e o valor corrente de Z.
    matriz = np.hstack([a, b.reshape(-1, 1)])
    matriz = np.vstack([matriz, np.append(r.copy(), z)])

    return PassoSimplex(
        matriz=matriz,
        colunas=colunas,
        rotulos_linhas=[colunas[bi] for bi in base],
        fase=reg["fase"],
        iteracao=reg["iter"],
        texto=reg["texto"],
        solucao=_solucao_atual(b, base, colunas, z),
        destaque_coluna=reg.get("destaque_coluna"),
        destaque_linha=reg.get("destaque_linha"),
        destaque_pivo=reg.get("destaque_pivo"),
        otimo=reg.get("otimo", False),
        final=reg.get("final", False),
        status=reg.get("status", STATUS_OTIMO),
    )


def _gerar_passos(registros: list[dict]) -> list[PassoSimplex]:
    """Expande registros em passos finos: coluna → linha → pivô → novo tableau."""
    if not registros:
        return []

    passos = [_formatar_passo(_passo_com_destaque(registros[0], {"inicial": True}))]

    for k, reg in enumerate(registros):
        col = reg.get("col")
        linha = reg.get("linha")
        if col is None:
            continue

        # Três destaques progressivos sobre o MESMO tableau (antes do pivô).
        passos.append(_formatar_passo(_passo_com_destaque(reg, {"destaque_coluna": col})))
        passos.append(_formatar_passo(
            _passo_com_destaque(reg, {"destaque_coluna": col, "destaque_linha": linha})
        ))
        passos.append(_formatar_passo(
            _passo_com_destaque(
                reg,
                {"destaque_coluna": col, "destaque_linha": linha, "destaque_pivo": (linha, col)},
            )
        ))

        # Novo tableau recalculado (o registro seguinte).
        if k + 1 < len(registros):
            passos.append(
                _formatar_passo(_passo_com_destaque(registros[k + 1], {"novo": True}))
            )

    # Passo final: destaque do ótimo/ilimitado sobre o último registro.
    passos.append(_formatar_passo(_passo_com_destaque(registros[-1], {"final_destaque": True})))
    return passos


# ------------------------------------------------------------------ solver principal
def resolver_simplex(
    c_in: list[float],
    tipo: str,
    restricoes: list[tuple[list[float], str, float]],
) -> ResultadoSimplex:
    """Resolve um PL de N variáveis pelo Simplex (Duas Fases).

    Parâmetros:
    - `c_in`: coeficientes do objetivo (comprimento N).
    - `tipo`: "max" ou "min".
    - `restricoes`: lista de triplas `(a, operador, b)` — `a` vetor de
      coeficientes (comprimento N), `operador` ∈ {"<=",">=","="}, `b` o RHS.
    """
    n = len(c_in)

    minimizar = tipo == "min"
    custo_real = [float(ci) for ci in c_in]
    custo = [-ci for ci in custo_real] if minimizar else custo_real.copy()

    # ---- Sem restrições: só o objetivo livre (x >= 0 assumido). --------------
    if not restricoes:
        if any(abs(ci) > EPS for ci in custo_real):
            return ResultadoSimplex(
                status=STATUS_ILIMITADO,
                mensagem="Problema ilimitado: objetivo não nulo e nenhuma restrição.",
            )
        return ResultadoSimplex(
            status=STATUS_OTIMO,
            z_otimo=0.0,
            solucao={f"x{i+1}": 0.0 for i in range(n)},
            mensagem="Solução ótima: Z = 0 (objetivo nulo).",
        )

    # ---- Normaliza restrições (garante b >= 0). ------------------------------
    restricoes_norm: list[tuple[list[float], str, float]] = []
    for a, op, b in restricoes:
        a = [float(x) for x in a]
        b = float(b)
        if len(a) != n:
            raise ValueError(f"Restrição com {len(a)} coeficientes, mas o objetivo tem {n}.")
        if b < 0:
            a = [-x for x in a]
            b = -b
            op = {"<=": ">=", ">=": "<="}.get(op, op)
        restricoes_norm.append((a, op, b))

    # ---- Monta variáveis (originais + folgas/excesso + artificiais). ----------
    # Primeiro determina TODAS as colunas (cada restrição pode acrescentar 1 ou 2
    # variáveis), e só depois monta as linhas com o comprimento total — assim as
    # linhas ficam homogêneas para virar um array numpy.
    colunas = [f"x{i+1}" for i in range(n)]
    custo_total = list(custo)
    extras_por_linha: list[dict[int, float]] = []
    rhs: list[float] = []
    base: list[int] = []

    cont_s = 0
    cont_a = 0
    for a, op, b in restricoes_norm:
        extras: dict[int, float] = {}
        if op == "<=":
            cont_s += 1
            colunas.append(f"s{cont_s}")
            custo_total.append(0.0)
            extras[len(colunas) - 1] = 1.0
        elif op == ">=":
            cont_s += 1
            colunas.append(f"s{cont_s}")  # variável de excesso (coef. -1)
            custo_total.append(0.0)
            extras[len(colunas) - 1] = -1.0
            cont_a += 1
            colunas.append(f"a{cont_a}")
            custo_total.append(0.0)
            extras[len(colunas) - 1] = 1.0
        else:  # "="
            cont_a += 1
            colunas.append(f"a{cont_a}")
            custo_total.append(0.0)
            extras[len(colunas) - 1] = 1.0
        base.append(len(colunas) - 1)
        extras_por_linha.append(extras)
        rhs.append(b)

    # Monta as linhas completas: coeficientes originais + slacks/excessos/artifs.
    num_vars = len(colunas)
    linhas: list[list[float]] = []
    for (a, _op, _b), extras in zip(restricoes_norm, extras_por_linha):
        linha = [0.0] * num_vars
        for j in range(n):
            linha[j] = a[j]
        for j, coef in extras.items():
            linha[j] = coef
        linhas.append(linha)

    a_mat = np.array(linhas, dtype=float)
    b_vec = np.array(rhs, dtype=float)
    custo_total = np.array(custo_total, dtype=float)

    registros: list[dict] = []

    # ---- Fase 1 (se houver artificiais). -------------------------------------
    tem_artificial = any(lab.startswith("a") for lab in colunas)
    if tem_artificial:
        custo_fase1 = np.array(
            [-1.0 if lab.startswith("a") else 0.0 for lab in colunas], dtype=float
        )
        _, a_mat, b_vec, base, _ = _executar_fase(
            a_mat, b_vec, custo_fase1, base, colunas, 1, registros
        )
        # Soma das artificiais = -z (pois o custo da Fase 1 é -1 por artificial).
        soma_artificiais = -float(_reducidos(a_mat, b_vec, custo_fase1, base)[1])
        if soma_artificiais > EPS:
            return ResultadoSimplex(
                status=STATUS_INVIAVEL,
                mensagem="Problema inviável: a Fase 1 não conseguiu zerar as variáveis artificiais.",
                passos=_gerar_passos(registros),
            )
        a_mat, b_vec, custo_total, colunas, base = _preparar_fase2(
            a_mat, b_vec, custo_total, colunas, base
        )

    # ---- Fase 2 (objetivo real). ---------------------------------------------
    _, a_mat, b_vec, base, status_f2 = _executar_fase(
        a_mat, b_vec, custo_total, base, colunas, 2, registros
    )

    r_final, z_final = _reducidos(a_mat, b_vec, custo_total, base)
    z_otimo = -z_final if minimizar else z_final
    solucao_final = _solucao_atual(b_vec, base, colunas, z_final)["x"]

    if status_f2 == STATUS_ILIMITADO:
        return ResultadoSimplex(
            status=STATUS_ILIMITADO,
            mensagem="Problema ilimitado: a coluna pivô não tem entradas positivas na razão mínima.",
            passos=_gerar_passos(registros),
        )

    return ResultadoSimplex(
        status=STATUS_OTIMO,
        z_otimo=z_otimo,
        solucao=solucao_final,
        mensagem=f"Solução ótima encontrada: Z = {z_otimo:.6f}.",
        passos=_gerar_passos(registros),
    )
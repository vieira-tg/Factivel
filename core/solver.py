"""Orquestração do método gráfico (core do solver).

Pipeline completo, passo a passo (ver também ARCHITECTURE.md):

1. Montar a lista de restrições ativas (incluindo x1>=0 e x2>=0 se pedido).
2. Para cada PAR de restrições, achar a interseção das duas retas.
3. Filtrar as interseções que satisfazem TODAS as restrições (vértices viáveis).
4. Remover duplicados (degenerescência) e ordenar em sentido anti-horário.
5. Avaliar Z em cada vértice.
6. Selecionar o(s) vértice(s) ótimo(s) conforme max/min.
7. Classificar os casos especiais: inviável, ilimitado, múltiplas soluções.

Para a classificação de inviável/ilimitado usamos o `scipy.optimize.linprog` como
segunda fonte de verdade. Motivo (registrado em CURRENT_TASK.md): "não há vértices"
NÃO equivale a "região inviável" — uma região viável com uma única restrição
(um semiplano não limitado) é viável mas não tem nenhum par de retas para gerar
vértice. O linprog distingue esses casos de forma exata.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import linprog

from .geometria import (
    TOL,
    eh_viavel,
    intersecao_retas,
    ordenar_anti_horario,
    remover_duplicados,
)
from .modelo import (
    OP_MAX,
    OP_MIN,
    STATUS_ILIMITADO,
    STATUS_INVIAVEL,
    STATUS_MULTIPLAS,
    STATUS_OTIMO,
    FuncaoObjetivo,
    Restricao,
    Resultado,
    Vertice,
)

# Tolerância para comparar valores de Z entre si (relativa à grandeza do Z).
_TOL_Z_RELATIVA = 1e-6


def _restricoes_ativas(
    restricoes: list[Restricao],
    incluir_nao_negatividade: bool,
) -> list[Restricao]:
    """Copia as restrições do usuário e, opcionalmente, acrescenta x1>=0 e x2>=0.

    As condições de não-negatividade são representadas como restrições comuns, de
    modo que também participam do cálculo de interseções (são elas que definem os
    eixos como arestas da região viável).
    """
    ativas = list(restricoes)
    if incluir_nao_negatividade:
        ativas.append(Restricao(1.0, 0.0, ">=", 0.0))  # x1 >= 0
        ativas.append(Restricao(0.0, 1.0, ">=", 0.0))  # x2 >= 0
    return ativas


def _linprog_status(
    objetivo: FuncaoObjetivo,
    ativas: list[Restricao],
) -> tuple[str, float | None]:
    """Resolve o mesmo problema com scipy.optimize.linprog (segunda fonte de verdade).

    Devolve `(status, z_otimo)` onde status é um dos STATUS_* e z_otimo é o valor
    ótimo quando o problema é limitado. O linprog minimiza por padrão; para
    maximizar, minimizamos -Z.
    """
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    a_eq: list[list[float]] = []
    b_eq: list[float] = []

    # Converte cada restrição para a forma canônica do linprog (<= ).
    for r in ativas:
        if r.operador == "<=":
            a_ub.append([r.a1, r.a2])
            b_ub.append(r.b)
        elif r.operador == ">=":
            # a*x >= b  =>  -a*x <= -b
            a_ub.append([-r.a1, -r.a2])
            b_ub.append(-r.b)
        else:  # "="
            a_eq.append([r.a1, r.a2])
            b_eq.append(r.b)

    # Vetor de custos: maximizar Z é o mesmo que minimizar -Z.
    c = [objetivo.c1, objetivo.c2]
    if objetivo.tipo == OP_MAX:
        c = [-objetivo.c1, -objetivo.c2]

    # Bounds explícitos são obrigatórios: com `bounds=None` (padrão), o HiGHS
    # pode interpretar as variáveis de forma inconsistente e reportar "inviável"
    # para problemas com `b` negativo e variáveis livres. Passar (None, None)
    # deixa claro que x1 e x2 são irrestritas por aqui (a não-negatividade, se
    # houver, já está contida nas linhas de A_ub).
    opt = linprog(
        c,
        A_ub=np.array(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(a_eq) if a_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=[(None, None), (None, None)],
        method="highs",
    )

    # status do linprog: 0 = ótimo, 2 = inviável, 3 = ilimitado.
    if opt.status == 2:
        return STATUS_INVIAVEL, None
    if opt.status == 3:
        return STATUS_ILIMITADO, None
    if opt.status == 0:
        # Recupera o sinal original de Z (vide vetor c acima).
        z = -opt.fun if objetivo.tipo == OP_MAX else opt.fun
        return STATUS_OTIMO, float(z)

    # status 1 (limite de iterações) ou 4 (problemas numéricos):
    # tratamos como inviável por segurança, mas marcamos na mensagem.
    return STATUS_INVIAVEL, None


def resolver(
    objetivo: FuncaoObjetivo,
    restricoes: list[Restricao],
    incluir_nao_negatividade: bool = True,
) -> Resultado:
    """Resolve o problema de PL pelo método gráfico e devolve o Resultado completo."""
    ativas = _restricoes_ativas(restricoes, incluir_nao_negatividade)

    # ---- Passo 1: gerar candidatos (interseção de cada par de retas). ---------
    candidatos: list[tuple[float, float]] = []
    for r1, r2 in combinations(ativas, 2):
        ponto = intersecao_retas(r1, r2)
        if ponto is None:
            continue  # retas paralelas/coincidentes: sem cruzamento único
        # ---- Passo 2: só é vértice se satisfizer TODAS as restrições. ---------
        if eh_viavel(ponto[0], ponto[1], ativas):
            candidatos.append(ponto)

    # ---- Passo 3: remover duplicados e ordenar o polígono. --------------------
    candidatos = remover_duplicados(candidatos)
    ordenados = ordenar_anti_horario(candidatos)
    vertices = [Vertice(x1=p[0], x2=p[1]) for p in ordenados]

    # ---- Passo 4: classificar status com ajuda do linprog. --------------------
    status_lp, z_lp = _linprog_status(objetivo, ativas)

    if status_lp == STATUS_INVIAVEL:
        return Resultado(
            status=STATUS_INVIAVEL,
            vertices=vertices,
            mensagem="Região inviável: nenhum ponto satisfaz todas as restrições.",
        )

    if status_lp == STATUS_ILIMITADO:
        direcao = "crescimento" if objetivo.tipo == OP_MAX else "decrescimento"
        return Resultado(
            status=STATUS_ILIMITADO,
            vertices=vertices,
            mensagem=(
                f"Problema ilimitado: a região viável se estende ao infinito na "
                f"direção de {direcao} de Z, portanto não há solução ótima finita."
            ),
        )

    # ---- Passo 5: avaliar Z em cada vértice. ----------------------------------
    for v in vertices:
        v.z = objetivo.avaliar(v.x1, v.x2)

    # ---- Passo 6: selecionar o(s) ótimo(s). -----------------------------------
    if vertices:
        z_otimo = (
            max(v.z for v in vertices if v.z is not None)
            if objetivo.tipo == OP_MAX
            else min(v.z for v in vertices if v.z is not None)
        )
        tol_z = _TOL_Z_RELATIVA * max(1.0, abs(z_otimo))
        vertices_otimos = [
            v for v in vertices if v.z is not None and abs(v.z - z_otimo) <= tol_z
        ]
    else:
        # Região viável mas sem vértices (ex.: um único semiplano). O ótimo
        # existe (linprog confirmou) mas não há vértice para exibir.
        z_otimo = z_lp
        vertices_otimos = []

    # ---- Passo 7: múltiplas soluções quando 2+ vértices empatam em Z*. --------
    multiplas = len(vertices_otimos) >= 2
    status = STATUS_MULTIPLAS if multiplas else STATUS_OTIMO

    if multiplas:
        mensagem = (
            f"Solução ótima múltipla: {len(vertices_otimos)} vértices atingem "
            f"Z* = {z_otimo:.4f}. A aresta (ou face) que os une é toda ela ótima."
        )
    else:
        mensagem = (
            f"Solução ótima encontrada em Z* = {z_otimo:.4f}."
        )

    return Resultado(
        status=status,
        vertices=vertices,
        vertices_otimos=vertices_otimos,
        z_otimo=z_otimo,
        mensagem=mensagem,
        z_linprog=z_lp,
    )
"""Primitivas geométricas do método gráfico.

Aqui ficam as três operações centrais do método gráfico em 2D:

1. **Interseção de duas retas**: resolver o sistema linear 2x2 formado por duas
   restrições trocadas por igualdade (`numpy.linalg.solve`).
2. **Teste de viabilidade**: verificar se um ponto satisfaz *todas* as restrições,
   respeitando o semiplano correto de cada uma.
3. **Ordenação angular**: dispor os vértices em sentido anti-horário ao redor do
   centróide para formar o polígono da região viável.
"""

from __future__ import annotations

import numpy as np

from .modelo import Restricao

# Tolerância numérica usada em todas as comparações. NUNCA comparar float com ==
# (regra do projeto): uma interseção pode cair em 2.0000000001 quando deveria
# valer 2.0, e um teste com `==` daria falso negativo.
TOL = 1e-9


def intersecao_retas(r1: Restricao, r2: Restricao) -> tuple[float, float] | None:
    """Calcula a interseção das retas definidas por r1 e r2 (operador trocado por `=`).

    Sistema 2x2:
        a1_1 * x1 + a2_1 * x2 = b_1
        a1_2 * x1 + a2_2 * x2 = b_2

    Se as retas forem paralelas (matriz singular) ou coincidentes, não existe um
    único ponto de interseção e devolvemos `None`.
    """
    a = np.array(
        [
            [r1.a1, r1.a2],
            [r2.a1, r2.a2],
        ],
        dtype=float,
    )
    b = np.array([r1.b, r2.b], dtype=float)

    # Uma matriz singular (determinante ~ 0) significa retas paralelas ou
    # coincidentes — logo não há um único ponto de cruzamento.
    if abs(np.linalg.det(a)) < TOL:
        return None

    solucao = np.linalg.solve(a, b)
    return (float(solucao[0]), float(solucao[1]))


def eh_viavel(
    x1: float,
    x2: float,
    restricoes: list[Restricao],
    tol: float = TOL,
) -> bool:
    """Verifica se (x1, x2) satisfaz TODAS as restrições.

    Cada restrição define um semiplano:

    - `<=`: o ponto deve ficar no lado menor-ou-igual da reta.
    - `>=`: o ponto deve ficar no lado maior-ou-igual da reta.
    - `=` : o ponto deve ficar exatamente sobre a reta.

    A tolerância evita descartar um vértice por erro de arredondamento de ponto
    flutuante (ex.: 2.0000000001 na verdade é 2.0).
    """
    for r in restricoes:
        esquerdo = r.avaliar_lado_esquerdo(x1, x2)
        if r.operador == "<=":
            if esquerdo > r.b + tol:  # esquerdo deve ser <= b
                return False
        elif r.operador == ">=":
            if esquerdo < r.b - tol:  # esquerdo deve ser >= b
                return False
        else:  # "="
            if abs(esquerdo - r.b) > tol:
                return False
    return True


def remover_duplicados(
    pontos: list[tuple[float, float]],
    tol: float = TOL,
) -> list[tuple[float, float]]:
    """Remove pontos repetidos (degenerescência: 3+ retas no mesmo ponto)."""
    unicos: list[tuple[float, float]] = []
    for p in pontos:
        # Só é duplicado se já existir um ponto "quase igual" na lista.
        if not any(
            abs(p[0] - q[0]) < tol and abs(p[1] - q[1]) < tol for q in unicos
        ):
            unicos.append(p)
    return unicos


def ordenar_anti_horario(
    pontos: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Ordena os vértices em sentido anti-horário ao redor do centróide.

    Funciona porque a região viável é um polígono convexo: ordenando pelo ângulo
    polar de cada vértice em relação ao centróide, os pontos ficam na ordem em que
    as arestas do polígono aparecem. Usa `atan2` para o ângulo em (-pi, pi].
    """
    if len(pontos) <= 2:
        return list(pontos)

    # Centróide = média das coordenadas, um ponto interno ao polígono convexo.
    cx = sum(p[0] for p in pontos) / len(pontos)
    cy = sum(p[1] for p in pontos) / len(pontos)

    def angulo(p: tuple[float, float]) -> float:
        return float(np.arctan2(p[1] - cy, p[0] - cx))

    return sorted(pontos, key=angulo)
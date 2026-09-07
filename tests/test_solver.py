"""Testes do motor de cálculo (core/solver.py).

Problemas clássicos de livro-texto, todos com solução conhecida, usados para
validar o método gráfico. Além do resultado esperado, cada teste compara o Z
ótimo com o valor retornado pelo `scipy.optimize.linprog` (segunda fonte de
verdade), garantindo que os dois métodos concordam.
"""

import math

from core.modelo import (
    STATUS_ILIMITADO,
    STATUS_INVIAVEL,
    STATUS_MULTIPLAS,
    STATUS_OTIMO,
    FuncaoObjetivo,
    Restricao,
)
from core.solver import resolver


def _aproximadamente(a, b, tol=1e-6):
    """Compara floats com tolerância (regra do projeto: nunca usar ==)."""
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_maximizar_classico_36():
    """Max Z = 3x1 + 5x2  s.a.  x1<=4, 2x2<=12, 3x1+2x2<=18, x1,x2>=0.

    Solução ótima conhecida: (x1, x2) = (2, 6) com Z = 36.
    """
    objetivo = FuncaoObjetivo(3, 5, "max")
    rest = [
        Restricao(1, 0, "<=", 4),
        Restricao(0, 2, "<=", 12),
        Restricao(3, 2, "<=", 18),
    ]
    r = resolver(objetivo, rest, incluir_nao_negatividade=True)

    assert r.status == STATUS_OTIMO
    assert len(r.vertices) == 5  # (0,0), (4,0), (4,3), (2,6), (0,6)
    assert len(r.vertices_otimos) == 1
    v = r.vertices_otimos[0]
    assert _aproximadamente(v.x1, 2)
    assert _aproximadamente(v.x2, 6)
    assert _aproximadamente(r.z_otimo, 36)
    # Validação cruzada com linprog.
    assert r.z_linprog is not None
    assert _aproximadamente(r.z_linprog, 36)


def test_minimizar_classico():
    """Min Z = 2x1 + 3x2  s.a.  x1+x2>=5, x1<=4, x2<=5, x1,x2>=0.

    Solução ótima conhecida: (x1, x2) = (4, 1) com Z = 11.
    """
    objetivo = FuncaoObjetivo(2, 3, "min")
    rest = [
        Restricao(1, 1, ">=", 5),
        Restricao(1, 0, "<=", 4),
        Restricao(0, 1, "<=", 5),
    ]
    r = resolver(objetivo, rest, incluir_nao_negatividade=True)

    assert r.status == STATUS_OTIMO
    v = r.vertices_otimos[0]
    assert _aproximadamente(v.x1, 4)
    assert _aproximadamente(v.x2, 1)
    assert _aproximadamente(r.z_otimo, 11)
    assert r.z_linprog is None or _aproximadamente(r.z_linprog, 11)


def test_multiplas_solucoes():
    """Max Z = x1 + x2  s.a.  x1+x2<=4, x1,x2>=0.

    A reta objetivo é paralela à aresta x1+x2=4; portanto a aresta inteira de
    (4,0) até (0,4) é ótima → múltiplas soluções.
    """
    objetivo = FuncaoObjetivo(1, 1, "max")
    rest = [Restricao(1, 1, "<=", 4)]
    r = resolver(objetivo, rest, incluir_nao_negatividade=True)

    assert r.status == STATUS_MULTIPLAS
    assert len(r.vertices_otimos) == 2  # (4,0) e (0,4)
    assert _aproximadamente(r.z_otimo, 4)
    assert r.z_linprog is None or _aproximadamente(r.z_linprog, 4)


def test_ilimitado():
    """Max Z = x1 + x2  s.a.  x1>=2, x2>=1. Sem limites superiores → ilimitado."""
    objetivo = FuncaoObjetivo(1, 1, "max")
    rest = [
        Restricao(1, 0, ">=", 2),
        Restricao(0, 1, ">=", 1),
    ]
    r = resolver(objetivo, rest, incluir_nao_negatividade=True)

    assert r.status == STATUS_ILIMITADO
    assert r.z_otimo is None


def test_inviavel():
    """Max Z = x1 + x2  s.a.  x1+x2<=2 E x1+x2>=5 (contradição) → inviável."""
    objetivo = FuncaoObjetivo(1, 1, "max")
    rest = [
        Restricao(1, 1, "<=", 2),
        Restricao(1, 1, ">=", 5),
    ]
    r = resolver(objetivo, rest, incluir_nao_negatividade=True)

    assert r.status == STATUS_INVIAVEL
    assert r.z_otimo is None


def test_sem_nao_negatividade():
    """Sem x1,x2>=0, a região e os vértices podem cair fora do 1º quadrante.

    Max Z = x1 + x2  s.a.  -2<=x1<=-1  e  -2<=x2<=-1  (quadrado todo negativo).
    Ótimo em (x1, x2) = (-1, -1) com Z = -2.
    """
    objetivo = FuncaoObjetivo(1, 1, "max")
    rest = [
        Restricao(1, 0, ">=", -2),
        Restricao(0, 1, ">=", -2),
        Restricao(1, 0, "<=", -1),
        Restricao(0, 1, "<=", -1),
    ]
    r = resolver(objetivo, rest, incluir_nao_negatividade=False)

    assert r.status == STATUS_OTIMO
    assert len(r.vertices) == 4
    v = r.vertices_otimos[0]
    assert _aproximadamente(v.x1, -1)
    assert _aproximadamente(v.x2, -1)
    assert _aproximadamente(r.z_otimo, -2)


def test_degenerescencia():
    """Problema com degenerescência (várias retas cruzando no mesmo ponto).

    Max Z = x1 + x2 s.a. x1 + x2 <= 4, 2*x1 + x2 <= 6, x1 + 2*x2 <= 6, x1,x2 >= 0.
    As retas se cruzam gerando vértices, testando remoção de duplicados.
    """
    objetivo = FuncaoObjetivo(1, 1, "max")
    rest = [
        Restricao(1, 1, "<=", 4),
        Restricao(2, 1, "<=", 6),
        Restricao(1, 2, "<=", 6),
    ]
    r = resolver(objetivo, rest, incluir_nao_negatividade=True)
    assert r.status == STATUS_OTIMO
    assert r.z_otimo is not None
    assert _aproximadamente(r.z_otimo, 4.0)  # em (2,2)
    assert r.z_linprog is not None
    assert _aproximadamente(r.z_linprog, r.z_otimo)


def test_coeficientes_negativos_objetivo():
    """Função objetivo com coeficientes negativos: Max Z = -2*x1 + 3*x2."""
    objetivo = FuncaoObjetivo(-2, 3, "max")
    rest = [
        Restricao(1, 0, "<=", 4),
        Restricao(0, 1, "<=", 5),
    ]
    r = resolver(objetivo, rest, incluir_nao_negatividade=True)
    assert r.status == STATUS_OTIMO
    assert r.z_linprog is not None
    assert _aproximadamente(r.z_otimo, r.z_linprog)


def test_multiplas_restricoes_mistas():
    """Mistura de restrições <=, >= e =."""
    objetivo = FuncaoObjetivo(3, 4, "max")
    rest = [
        Restricao(1, 1, ">=", 2),
        Restricao(1, 2, "<=", 10),
        Restricao(1, 0, "<=", 6),
    ]
    r = resolver(objetivo, rest, incluir_nao_negatividade=True)
    assert r.status == STATUS_OTIMO
    assert r.z_linprog is not None
    assert _aproximadamente(r.z_otimo, r.z_linprog)

"""Testes do método Simplex (core/simplex.py).

Problemas de PL variados (2 e 3 variáveis, max/min, <=, >=, =, inviável e
ilimitado), sempre comparando o resultado com o `scipy.optimize.linprog` como
segunda fonte de verdade — o mesmo padrão dos testes do gráfico.
"""

import numpy as np
from scipy.optimize import linprog

from core.modelo import STATUS_ILIMITADO, STATUS_INVIAVEL, STATUS_OTIMO
from core.simplex import resolver_simplex


def _z_linprog(c, tipo, restricoes):
    """Mesmo problema resolvido pelo linprog (variáveis >= 0)."""
    a_ub, b_ub, a_eq, b_eq = [], [], [], []
    for a, op, b in restricoes:
        if op == "<=":
            a_ub.append(a)
            b_ub.append(b)
        elif op == ">=":
            a_ub.append([-x for x in a])
            b_ub.append(-b)
        else:
            a_eq.append(a)
            b_eq.append(b)
    custo = c if tipo == "min" else [-ci for ci in c]
    r = linprog(
        custo,
        A_ub=np.array(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(a_eq) if a_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=[(0, None)] * len(c),
        method="highs",
    )
    return r.status, (None if r.fun is None else (-r.fun if tipo == "max" else r.fun))


def _aprox(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_maximizar_2_variaveis():
    """Max 3x1 + 2x2  s.a.  x1+2x2<=8, 3x1+x2<=9  =>  (2, 3), Z=12."""
    res = resolver_simplex([3, 2], "max", [([1, 2], "<=", 8), ([3, 1], "<=", 9)])

    assert res.status == STATUS_OTIMO
    assert _aprox(res.z_otimo, 12)
    assert _aprox(res.solucao["x1"], 2)
    assert _aprox(res.solucao["x2"], 3)
    # Cada pivô gera passos (destaques + tableau novo) e há um passo final ótimo.
    assert len(res.passos) > 0
    assert res.passos[-1].otimo is True


def test_maximizar_3_variaveis():
    """Max 2x1 + 3x2 + 4x3  s.a.  x1+x2+x3<=5, 2x1+x3<=7, x2+2x3<=8."""
    c = [2, 3, 4]
    rest = [([1, 1, 1], "<=", 5), ([2, 0, 1], "<=", 7), ([0, 1, 2], "<=", 8)]
    res = resolver_simplex(c, "max", rest)
    assert res.status == STATUS_OTIMO
    assert _aprox(res.z_otimo, 18.0)


def test_restricao_maior_igual():
    """Max 2x1 + 3x2  s.a.  x1+x2>=4, x1<=3, x2<=3  =>  (3, 3), Z=15 (duas fases)."""
    rest = [([1, 1], ">=", 4), ([1, 0], "<=", 3), ([0, 1], "<=", 3)]
    res = resolver_simplex([2, 3], "max", rest)

    assert res.status == STATUS_OTIMO
    assert _aprox(res.z_otimo, 15)
    assert _aprox(res.solucao["x1"], 3)
    assert _aprox(res.solucao["x2"], 3)


def test_minimizar():
    """Min 2x1 + 3x2  s.a.  x1+x2<=10, x1>=2  =>  (2, 0), Z=4."""
    rest = [([1, 1], "<=", 10), ([1, 0], ">=", 2)]
    res = resolver_simplex([2, 3], "min", rest)

    assert res.status == STATUS_OTIMO
    assert _aprox(res.z_otimo, 4)
    assert _aprox(res.solucao["x1"], 2)
    assert _aprox(res.solucao["x2"], 0)


def test_restricao_igualdade():
    """Max x1 + x2  s.a.  x1 + x2 = 4  =>  Z = 4."""
    res = resolver_simplex([1, 1], "max", [([1, 1], "=", 4)])
    assert res.status == STATUS_OTIMO
    assert _aprox(res.z_otimo, 4)


def test_inviavel():
    """Max x1+x2 s.a. x1+x2<=2 e x1+x2<=-5 (contradição) => inviável."""
    rest = [([1, 1], "<=", 2), ([1, 1], "<=", -5)]
    res = resolver_simplex([1, 1], "max", rest)
    assert res.status == STATUS_INVIAVEL
    assert res.z_otimo is None


def test_ilimitado():
    """Max x1+x2 s.a. x1+x2>=2 => ilimitado."""
    res = resolver_simplex([1, 1], "max", [([1, 1], ">=", 2)])
    assert res.status == STATUS_ILIMITADO
    assert res.z_otimo is None


def test_compara_linprog_varios():
    """Valida o Z ótimo contra o linprog em vários problemas."""
    problemas = [
        ([3, 2], "max", [([1, 2], "<=", 8), ([3, 1], "<=", 9)]),
        ([2, 3, 4], "max", [([1, 1, 1], "<=", 5), ([2, 0, 1], "<=", 7), ([0, 1, 2], "<=", 8)]),
        ([2, 3], "max", [([1, 1], ">=", 4), ([1, 0], "<=", 3), ([0, 1], "<=", 3)]),
        ([2, 3], "min", [([1, 1], "<=", 10), ([1, 0], ">=", 2)]),
        ([1, 2, 3, 4], "max", [([1, 0, 0, 0], "<=", 5), ([0, 1, 0, 0], "<=", 5),
                                ([0, 0, 1, 0], "<=", 5), ([0, 0, 0, 1], "<=", 5)]),
    ]
    for c, tipo, rest in problemas:
        st_ref, z_ref = _z_linprog(c, tipo, rest)
        res = resolver_simplex(c, tipo, rest)
        if st_ref == 0:
            assert res.status == STATUS_OTIMO
            assert _aprox(res.z_otimo, z_ref)


def test_simplex_minimizacao_com_igualdades():
    """Minimização com restrições de igualdade e desigualdade."""
    c = [1, 2, 3]
    rest = [
        ([1, 1, 1], "=", 6),
        ([2, -1, 1], ">=", 2),
    ]
    st_ref, z_ref = _z_linprog(c, "min", rest)
    res = resolver_simplex(c, "min", rest)
    if st_ref == 0:
        assert res.status == STATUS_OTIMO
        assert _aprox(res.z_otimo, z_ref)


def test_simplex_caso_degenerado_ou_multiplo():
    """Problema simplex com restrições adicionais."""
    c = [4, 3]
    rest = [
        ([2, 1], "<=", 10),
        ([1, 1], "<=", 8),
        ([1, 2], "<=", 12),
    ]
    st_ref, z_ref = _z_linprog(c, "max", rest)
    res = resolver_simplex(c, "max", rest)
    assert res.status == STATUS_OTIMO
    assert _aprox(res.z_otimo, z_ref)

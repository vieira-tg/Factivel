"""Modelo de dados do problema de Programação Linear.

Este módulo contém apenas estruturas de dados (dataclasses) que descrevem o
problema e a solução. Nenhuma lógica de cálculo fica aqui — a matemática do
método gráfico está em `geometria.py` e `solver.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Operadores aceitos em uma restrição. `<=` e `>=` definem semiplanos; `=` define
# uma reta que obriga o ponto a pertencer exatamente a ela.
OP_MAX = "max"
OP_MIN = "min"
OPERADORES_VALIDOS = ("<=", ">=", "=")


@dataclass(frozen=True)
class FuncaoObjetivo:
    """Função objetivo: Z = c1*x1 + c2*x2, a ser maximizada ou minimizada."""

    c1: float
    c2: float
    tipo: str = OP_MAX  # "max" ou "min"

    def avaliar(self, x1: float, x2: float) -> float:
        """Calcula o valor de Z em um ponto (x1, x2)."""
        return self.c1 * x1 + self.c2 * x2


@dataclass(frozen=True)
class Restricao:
    """Inequação linear: a1*x1 + a2*x2 {<=, >=, =} b."""

    a1: float
    a2: float
    operador: str  # "<=", ">=" ou "="
    b: float

    def avaliar_lado_esquerdo(self, x1: float, x2: float) -> float:
        """Calcula o valor de a1*x1 + a2*x2 no ponto (x1, x2)."""
        return self.a1 * x1 + self.a2 * x2

    def __str__(self) -> str:
        return f"{self.a1:g}*x1 + {self.a2:g}*x2 {self.operador} {self.b:g}"


@dataclass
class Vertice:
    """Ponto extremo da região viável (interseção de duas retas de restrição)."""

    x1: float
    x2: float
    z: float | None = None  # valor da função objetivo neste vértice

    def coords(self) -> tuple[float, float]:
        return (self.x1, self.x2)


# Status possíveis de um problema resolvido.
STATUS_OTIMO = "otimo"
STATUS_MULTIPLAS = "multiplas"
STATUS_INVIAVEL = "inviavel"
STATUS_ILIMITADO = "ilimitado"


@dataclass
class Resultado:
    """Resultado completo da resolução de um problema de PL.

    - `vertices`: vértices viáveis ordenados em sentido anti-horário.
    - `vertices_otimos`: vértices que atingem o valor ótimo (1 ou mais).
    - `z_otimo`: valor ótimo de Z (None quando inviável/ilimitado).
    - `status`: um dos STATUS_* acima.
    - `mensagem`: texto amigável (pt-BR) para a interface.
    - `z_linprog`: valor ótimo retornado pelo scipy.optimize.linprog (validação
      cruzada); None quando não aplicável.
    """

    status: str = STATUS_OTIMO
    vertices: list[Vertice] = field(default_factory=list)
    vertices_otimos: list[Vertice] = field(default_factory=list)
    z_otimo: float | None = None
    mensagem: str = ""
    z_linprog: float | None = None
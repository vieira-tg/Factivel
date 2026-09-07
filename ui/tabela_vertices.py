"""Tabela de vértices e respectivos valores de Z.

Apresenta cada vértice viável com as coordenadas (x1, x2) e o valor de Z da
função objetivo. Os vértices ótimos são destacados em destaque (fundo verde) e
rotulados com a palavra "ótimo".
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

from core.modelo import Resultado


def _formatar(v: float) -> str:
    """Formata um float enxuto (até 4 casas, sem zeros excedentes)."""
    return f"{v:.4g}"


class TabelaVertices(QTableWidget):
    """QTableWidget com colunas Vértice / x1 / x2 / Z."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Vértice", "x1", "x2", "Z"])
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.horizontalHeader().setStretchLastSection(True)

    def limpar(self) -> None:
        self.setRowCount(0)

    def atualizar(self, resultado: Resultado) -> None:
        """Preenche a tabela com os vértices e destaca os ótimos."""
        self.limpar()

        otimos = resultado.vertices_otimos if resultado.vertices_otimos else []
        ids_otimos = set(id(v) for v in otimos)

        for i, v in enumerate(resultado.vertices, start=1):
            self.insertRow(self.rowCount())
            linha = self.rowCount() - 1

            # Coluna 0: rótulo "V1", "V2", ... (com "*" e "ótimo" se for ótimo).
            eh_otimo = id(v) in ids_otimos
            rotulo = f"V{i}" + (" (ótimo)" if eh_otimo else "")
            item_rotulo = QTableWidgetItem(rotulo)

            itens = [
                item_rotulo,
                QTableWidgetItem(_formatar(v.x1)),
                QTableWidgetItem(_formatar(v.x2)),
                QTableWidgetItem(_formatar(v.z) if v.z is not None else "—"),
            ]

            for col, item in enumerate(itens):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setItem(linha, col, item)

            if eh_otimo:
                # Destaca a linha inteira do(s) vértice(s) ótimo(s).
                fonte = QFont()
                fonte.setBold(True)
                for col in range(self.columnCount()):
                    self.item(linha, col).setBackground(QColor("#c8e6c9"))
                    self.item(linha, col).setFont(fonte)
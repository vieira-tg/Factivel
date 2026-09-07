"""Ponto de entrada do Factível.

Inicializa a QApplication e exibe a janela principal. Execute com:

    python main.py

(a partir da pasta `Factivel/`, que funciona como raiz do projeto).
"""

from __future__ import annotations

import os
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow

# Caminho do ícone (relativo a este arquivo, para funcionar de qualquer pasta).
_CAMINHO_ICONE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "recursos", "factivel.png"
)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Factível")
    app.setWindowIcon(QIcon(_CAMINHO_ICONE))

    janela = MainWindow()
    janela.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
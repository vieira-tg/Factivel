"""Janela principal: formulário de entrada + gráfico + tabela de vértices.

Layout em duas colunas:

- à esquerda, o formulário (função objetivo, lista dinâmica de restrições,
  opção de não-negatividade) e o botão "Resolver";
- à direita, o gráfico animado, um rótulo de status e a tabela de vértices.

O menu "Arquivo" oferece salvar/carregar o problema (JSON) e exportar o gráfico
como imagem (Fase 6 do roadmap).
"""

from __future__ import annotations

import json
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.modelo import (
    OP_MAX,
    OP_MIN,
    STATUS_ILIMITADO,
    STATUS_INVIAVEL,
    STATUS_MULTIPLAS,
    FuncaoObjetivo,
    Restricao,
)
from ui.plot_widget import PlotWidget
from ui.simplex_widget import AbaSimplex
from ui.tabela_vertices import TabelaVertices


def _spinbox(valor: float = 0.0) -> QDoubleSpinBox:
    """Cria um QDoubleSpinBox padrão para os coeficientes."""
    sp = QDoubleSpinBox()
    sp.setRange(-1e9, 1e9)
    sp.setDecimals(4)
    sp.setValue(valor)
    sp.setMinimumWidth(72)
    return sp


class ListaRestricoes(QWidget):
    """Lista dinâmica de restrições (adicionar/remover linhas)."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.tabela = QTableWidget(0, 4)
        self.tabela.setHorizontalHeaderLabels(["a1", "a2", "operador", "b"])
        self.tabela.verticalHeader().setVisible(False)
        self.tabela.horizontalHeader().setStretchLastSection(True)

        self.btn_adicionar = QPushButton("Adicionar restrição")
        self.btn_remover = QPushButton("Remover selecionada")
        self.btn_adicionar.clicked.connect(lambda: self.adicionar_linha())
        self.btn_remover.clicked.connect(self.remover_selecionada)

        botoes = QHBoxLayout()
        botoes.addWidget(self.btn_adicionar)
        botoes.addWidget(self.btn_remover)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabela)
        layout.addLayout(botoes)

    def adicionar_linha(
        self, a1: float = 0.0, a2: float = 0.0, operador: str = "<=", b: float = 0.0
    ) -> None:
        linha = self.tabela.rowCount()
        self.tabela.insertRow(linha)

        self.tabela.setCellWidget(linha, 0, _spinbox(a1))
        self.tabela.setCellWidget(linha, 1, _spinbox(a2))

        combo = QComboBox()
        combo.addItems(["<=", ">=", "="])
        combo.setCurrentText(operador)
        self.tabela.setCellWidget(linha, 2, combo)

        self.tabela.setCellWidget(linha, 3, _spinbox(b))

    def remover_selecionada(self) -> None:
        linha = self.tabela.currentRow()
        if linha >= 0:
            self.tabela.removeRow(linha)

    def obter_restricoes(self) -> list[Restricao]:
        restricoes: list[Restricao] = []
        for linha in range(self.tabela.rowCount()):
            a1 = self.tabela.cellWidget(linha, 0).value()
            a2 = self.tabela.cellWidget(linha, 1).value()
            operador = self.tabela.cellWidget(linha, 2).currentText()
            b = self.tabela.cellWidget(linha, 3).value()
            restricoes.append(Restricao(a1, a2, operador, b))
        return restricoes

    def carregar(self, restricoes: list[Restricao]) -> None:
        """Substitui a lista atual por um novo conjunto de restrições."""
        self.tabela.setRowCount(0)
        for r in restricoes:
            self.adicionar_linha(r.a1, r.a2, r.operador, r.b)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Factível")
        self.resize(1200, 760)

        # Ícone da janela (mesma arte usada como ícone do aplicativo).
        caminho_icone = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "recursos",
            "factivel.png",
        )
        self.setWindowIcon(QIcon(caminho_icone))

        # ------------------------- formulário (esquerda) ----------------------
        self.spin_c1 = _spinbox(3.0)
        self.spin_c2 = _spinbox(5.0)
        self.combo_tipo = QComboBox()
        self.combo_tipo.addItems(["max", "min"])
        self.combo_tipo.setCurrentText("max")

        grupo_objetivo = QGroupBox("Função objetivo  Z = c1·x1 + c2·x2")
        form_objetivo = QFormLayout(grupo_objetivo)
        form_objetivo.addRow("c1 (coef. de x1):", self.spin_c1)
        form_objetivo.addRow("c2 (coef. de x2):", self.spin_c2)
        form_objetivo.addRow("Objetivo:", self.combo_tipo)

        self.check_nn = QCheckBox("Incluir não-negatividade (x1 ≥ 0 e x2 ≥ 0)")
        self.check_nn.setChecked(True)

        grupo_restricoes = QGroupBox("Restrições")
        self.lista_restricoes = ListaRestricoes()
        layout_restricoes = QVBoxLayout(grupo_restricoes)
        layout_restricoes.addWidget(self.lista_restricoes)

        self.btn_resolver = QPushButton("Resolver")
        self.btn_resolver.setMinimumHeight(36)

        painel_esquerda = QWidget()
        layout_esquerda = QVBoxLayout(painel_esquerda)
        layout_esquerda.addWidget(grupo_objetivo)
        layout_esquerda.addWidget(self.check_nn)
        layout_esquerda.addWidget(grupo_restricoes, 1)
        layout_esquerda.addWidget(self.btn_resolver)

        # ------------------------- saída (direita) ----------------------------
        self.rotulo_status = QLabel("Informe os dados e clique em Resolver.")
        self.rotulo_status.setWordWrap(True)

        self.plot = PlotWidget()
        self.tabela = TabelaVertices()

        painel_direita = QWidget()
        layout_direita = QVBoxLayout(painel_direita)
        layout_direita.addWidget(self.rotulo_status)
        layout_direita.addWidget(self.plot, 1)
        layout_direita.addWidget(self.tabela, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(painel_esquerda)
        splitter.addWidget(painel_direita)
        splitter.setSizes([360, 840])

        # Aba do Simplex (N variáveis).
        aba_simplex = AbaSimplex()

        abas = QTabWidget()
        abas.addTab(splitter, "Método Gráfico")
        abas.addTab(aba_simplex, "Simplex")
        self.setCentralWidget(abas)

        # ------------------------- conexões -----------------------------------
        self.btn_resolver.clicked.connect(self.resolver)

        # ------------------------- menu (Fase 6) ------------------------------
        self._criar_menu()

        # Pré-carrega um problema clássico como exemplo visual.
        self._carregar_exemplo()

    # --------------------------------------------------------------- menu
    def _criar_menu(self) -> None:
        menu_arquivo = self.menuBar().addMenu("&Arquivo")

        acao_salvar = QAction("&Salvar problema...", self)
        acao_salvar.setShortcut("Ctrl+S")
        acao_salvar.triggered.connect(self.salvar_problema)
        menu_arquivo.addAction(acao_salvar)

        acao_carregar = QAction("&Carregar problema...", self)
        acao_carregar.setShortcut("Ctrl+O")
        acao_carregar.triggered.connect(self.carregar_problema)
        menu_arquivo.addAction(acao_carregar)

        menu_arquivo.addSeparator()

        acao_exportar = QAction("&Exportar gráfico (PNG)...", self)
        acao_exportar.triggered.connect(self.exportar_grafico)
        menu_arquivo.addAction(acao_exportar)

        menu_arquivo.addSeparator()

        acao_sair = QAction("Sai&r", self)
        acao_sair.triggered.connect(QApplication.instance().quit)
        menu_arquivo.addAction(acao_sair)

    # ------------------------------------------------------------ exemplo
    def _carregar_exemplo(self) -> None:
        """Problema clássico: Max Z = 3x1 + 5x2  s.a.  x1<=4, 2x2<=12, 3x1+2x2<=18."""
        self.lista_restricoes.adicionar_linha(1.0, 0.0, "<=", 4.0)
        self.lista_restricoes.adicionar_linha(0.0, 2.0, "<=", 12.0)
        self.lista_restricoes.adicionar_linha(3.0, 2.0, "<=", 18.0)

    # ------------------------------------------------------------ resolver
    def resolver(self) -> None:
        c1 = self.spin_c1.value()
        c2 = self.spin_c2.value()
        tipo = self.combo_tipo.currentText()
        objetivo = FuncaoObjetivo(c1, c2, tipo)

        restricoes = self.lista_restricoes.obter_restricoes()
        incluir_nn = self.check_nn.isChecked()

        resultado = self.plot.resolver_problema(objetivo, restricoes, incluir_nn)

        self.tabela.atualizar(resultado)
        self._atualizar_status(resultado)

    def _atualizar_status(self, resultado) -> None:
        """Mostra a mensagem de status, colorida conforme o caso."""
        cor = "#2e7d32"  # verde para ótimo
        if resultado.status == STATUS_INVIAVEL:
            cor = "#c62828"
        elif resultado.status == STATUS_ILIMITADO:
            cor = "#e65100"
        elif resultado.status == STATUS_MULTIPLAS:
            cor = "#1565c0"

        self.rotulo_status.setText(resultado.mensagem)
        self.rotulo_status.setStyleSheet(f"font-weight: bold; color: {cor};")

    # ---------------------------------------------- salvar / carregar / exportar
    def _dados_do_problema(self) -> dict:
        return {
            "objetivo": {
                "c1": self.spin_c1.value(),
                "c2": self.spin_c2.value(),
                "tipo": self.combo_tipo.currentText(),
            },
            "incluir_nao_negatividade": self.check_nn.isChecked(),
            "restricoes": [
                {
                    "a1": r.a1,
                    "a2": r.a2,
                    "operador": r.operador,
                    "b": r.b,
                }
                for r in self.lista_restricoes.obter_restricoes()
            ],
        }

    def salvar_problema(self) -> None:
        caminho, _ = QFileDialog.getSaveFileName(
            self, "Salvar problema", "", "Problema de PL (*.json)"
        )
        if not caminho:
            return
        if not caminho.lower().endswith(".json"):
            caminho += ".json"
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(self._dados_do_problema(), f, ensure_ascii=False, indent=2)

    def carregar_problema(self) -> None:
        caminho, _ = QFileDialog.getOpenFileName(
            self, "Carregar problema", "", "Problema de PL (*.json)"
        )
        if not caminho:
            return
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                dados = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            QMessageBox.critical(self, "Erro", f"Não foi possível ler o arquivo:\n{e}")
            return

        obj = dados.get("objetivo", {})
        self.spin_c1.setValue(float(obj.get("c1", 0)))
        self.spin_c2.setValue(float(obj.get("c2", 0)))
        self.combo_tipo.setCurrentText(
            obj.get("tipo", "max") if obj.get("tipo") in (OP_MAX, OP_MIN) else OP_MAX
        )
        self.check_nn.setChecked(bool(dados.get("incluir_nao_negatividade", True)))

        restricoes = dados.get("restricoes", [])
        lista = [
            Restricao(
                float(r.get("a1", 0)),
                float(r.get("a2", 0)),
                r.get("operador", "<="),
                float(r.get("b", 0)),
            )
            for r in restricoes
        ]
        self.lista_restricoes.carregar(lista)

    def exportar_grafico(self) -> None:
        caminho, _ = QFileDialog.getSaveFileName(
            self, "Exportar gráfico", "", "Imagem PNG (*.png)"
        )
        if not caminho:
            return
        if not caminho.lower().endswith(".png"):
            caminho += ".png"
        self.plot.figure.savefig(caminho, dpi=150)
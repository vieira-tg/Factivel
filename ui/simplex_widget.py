"""Aba do método Simplex (N variáveis) com tableaus animados.

Entrada: número de variáveis, coeficientes do objetivo (vetor), tipo (max/min) e
lista de restrições (cada uma com N coeficientes + operador + b).

Saída: o algoritmo produz uma sequência de tableaus (matriz numpy) com
destaques por passo — coluna pivô, linha pivô, elemento pivô — no mesmo espírito
da animação do gráfico, só que aqui o usuário controla o ritmo com os botões
"Próximo"/"Anterior" (modo manual) ou "Reproduzir" (modo automático via QTimer).
O "Pular" mostra direto a tabela final.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.modelo import STATUS_ILIMITADO, STATUS_INVIAVEL, STATUS_OTIMO
from core.simplex import PassoSimplex, ResultadoSimplex, resolver_simplex

# Cores dos destaques do tableau.
COR_COLUNA = "#bbdefb"   # azul claro: variável que ENTRA
COR_LINHA = "#ffe0b2"    # laranja claro: variável que SAI
COR_PIVO = "#d32f2f"     # vermelho: elemento pivô
COR_OTIMO = "#c8e6c9"    # verde: tabela ótima


def _fmt(v: float) -> str:
    """Formata um float enxuto para célula do tableau."""
    if v == 0:
        return "0"
    return f"{v:.4g}"


def _spinbox(valor: float = 0.0) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(-1e9, 1e9)
    sp.setDecimals(4)
    sp.setValue(valor)
    sp.setMinimumWidth(66)
    return sp


class AbaSimplex(QWidget):
    """Widget completo da aba Simplex."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._n = 2

        # ------------------------- entrada -----------------------------------
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 8)
        self.spin_n.setValue(2)
        self.spin_n.valueChanged.connect(self._mudou_numero_variaveis)

        self.combo_tipo = QComboBox()
        self.combo_tipo.addItems(["max", "min"])

        self.tabela_objetivo = QTableWidget(1, 0)
        self.tabela_objetivo.verticalHeader().setVisible(False)

        grupo_objetivo = QGroupBox("Função objetivo")
        linha_topo = QHBoxLayout()
        linha_topo.addWidget(QLabel("Nº de variáveis:"))
        linha_topo.addWidget(self.spin_n)
        linha_topo.addSpacing(12)
        linha_topo.addWidget(QLabel("Objetivo:"))
        linha_topo.addWidget(self.combo_tipo)
        linha_topo.addStretch(1)

        self.rotulo_z = QLabel("Z = ")

        layout_objetivo = QVBoxLayout(grupo_objetivo)
        layout_objetivo.addLayout(linha_topo)
        layout_objetivo.addWidget(QLabel("Coeficientes (c1, c2, ...):"))
        layout_objetivo.addWidget(self.tabela_objetivo)
        layout_objetivo.addWidget(self.rotulo_z)

        # Restrições
        self.tabela_restricoes = QTableWidget(0, 0)
        self.tabela_restricoes.verticalHeader().setVisible(False)

        self.btn_adicionar = QPushButton("Adicionar restrição")
        self.btn_remover = QPushButton("Remover selecionada")
        self.btn_adicionar.clicked.connect(self._adicionar_restricao)
        self.btn_remover.clicked.connect(self._remover_restricao)

        botoes_rest = QHBoxLayout()
        botoes_rest.addWidget(self.btn_adicionar)
        botoes_rest.addWidget(self.btn_remover)
        botoes_rest.addStretch(1)

        grupo_restricoes = QGroupBox("Restrições")
        layout_restricoes = QVBoxLayout(grupo_restricoes)
        layout_restricoes.addWidget(self.tabela_restricoes)
        layout_restricoes.addLayout(botoes_rest)

        # Exemplos rápidos
        linha_exemplos = QHBoxLayout()
        btn_ex2 = QPushButton("Exemplo 2 variáveis")
        btn_ex3 = QPushButton("Exemplo 3 variáveis")
        btn_ex2.clicked.connect(self._carregar_exemplo_2)
        btn_ex3.clicked.connect(self._carregar_exemplo_3)
        linha_exemplos.addWidget(btn_ex2)
        linha_exemplos.addWidget(btn_ex3)
        linha_exemplos.addStretch(1)

        self.btn_resolver = QPushButton("Resolver pelo Simplex")
        self.btn_resolver.setMinimumHeight(32)

        painel_entrada = QWidget()
        layout_entrada = QVBoxLayout(painel_entrada)
        layout_entrada.addWidget(grupo_objetivo)
        layout_entrada.addWidget(grupo_restricoes)
        layout_entrada.addLayout(linha_exemplos)
        layout_entrada.addWidget(self.btn_resolver)

        # ------------------------- saída --------------------------------------
        self.rotulo_status = QLabel("Informe os dados e clique em Resolver.")
        self.rotulo_status.setWordWrap(True)

        self.rotulo_passo = QLabel("")
        self.rotulo_passo.setWordWrap(True)

        self.tabela_simplex = QTableWidget(0, 0)
        self.tabela_simplex.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabela_simplex.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        self.rotulo_solucao = QLabel("")

        # Controles de navegação da animação
        self.btn_anterior = QPushButton("◀ Anterior")
        self.btn_proximo = QPushButton("Próximo ▶")
        self.btn_reproduzir = QPushButton("▶ Reproduzir")
        self.btn_pular = QPushButton("Pular (fim)")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 10)
        self.slider.setValue(5)

        self.btn_anterior.clicked.connect(self.anterior)
        self.btn_proximo.clicked.connect(self.proximo)
        self.btn_reproduzir.clicked.connect(self._alternar_reproducao)
        self.btn_pular.clicked.connect(self.pular_fim)

        linha_controles = QHBoxLayout()
        linha_controles.addWidget(self.btn_anterior)
        linha_controles.addWidget(self.btn_proximo)
        linha_controles.addWidget(self.btn_reproduzir)
        linha_controles.addWidget(self.btn_pular)
        linha_controles.addWidget(QLabel("Lento"))
        linha_controles.addWidget(self.slider)
        linha_controles.addWidget(QLabel("Rápido"))

        painel_saida = QWidget()
        layout_saida = QVBoxLayout(painel_saida)
        layout_saida.addWidget(self.rotulo_status)
        layout_saida.addWidget(self.rotulo_passo)
        layout_saida.addWidget(self.tabela_simplex, 1)
        layout_saida.addWidget(self.rotulo_solucao)
        layout_saida.addLayout(linha_controles)

        # ------------------------- timer da animação --------------------------
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        # Estado da animação
        self._passos: list[PassoSimplex] = []
        self._indice = -1
        self._reproduzindo = False

        # ------------------------- layout geral --------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(painel_entrada)
        splitter.addWidget(painel_saida)
        splitter.setSizes([380, 820])
        splitter.setChildrenCollapsible(False)

        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

        self.btn_resolver.clicked.connect(self.resolver)

        # Estado inicial: tabela de entrada do tamanho de n.
        self._reconstruir_tabelas([0.0] * self._n, [])
        self._carregar_exemplo_2()

    # ------------------------------------------------ entrada (N variáveis)
    def _mudou_numero_variaveis(self, novo_n: int) -> None:
        """Ajusta o nº de variáveis preservando os valores já digitados."""
        c_atual = self._ler_objetivo()
        restricoes_atual = self._ler_restricoes()

        c_novo = (c_atual + [0.0] * novo_n)[:novo_n]
        restricoes_novo = []
        for r in restricoes_atual:
            restricoes_novo.append(
                {
                    "a": (r["a"] + [0.0] * novo_n)[:novo_n],
                    "op": r["op"],
                    "b": r["b"],
                }
            )
        self._n = novo_n
        self._reconstruir_tabelas(c_novo, restricoes_novo)

    def _reconstruir_tabelas(self, c: list[float], restricoes: list[dict]) -> None:
        """Reconstrói as tabelas de entrada (objetivo e restrições) para o nº atual."""
        n = len(c)
        self._n = n

        # Tabela do objetivo: 1 linha x n colunas (x1..xn).
        self.tabela_objetivo.clear()
        self.tabela_objetivo.setRowCount(1)
        self.tabela_objetivo.setColumnCount(n)
        self.tabela_objetivo.setHorizontalHeaderLabels([f"x{i+1}" for i in range(n)])
        for j in range(n):
            self.tabela_objetivo.setCellWidget(0, j, _spinbox(c[j]))

        # Tabela de restrições: n colunas de coeficientes + operador + b.
        self.tabela_restricoes.clear()
        self.tabela_restricoes.setRowCount(0)
        self.tabela_restricoes.setColumnCount(n + 2)
        cabecalho = [f"x{i+1}" for i in range(n)] + ["operador", "b"]
        self.tabela_restricoes.setHorizontalHeaderLabels(cabecalho)
        for r in restricoes:
            self._adicionar_restricao(r["a"], r["op"], r["b"])

        self._atualizar_rotulo_z()

    def _adicionar_restricao(
        self, a: list[float] | None = None, op: str = "<=", b: float = 0.0
    ) -> None:
        linha = self.tabela_restricoes.rowCount()
        self.tabela_restricoes.insertRow(linha)
        n = self._n
        a = a or [0.0] * n
        for j in range(n):
            self.tabela_restricoes.setCellWidget(linha, j, _spinbox(a[j]))
        combo = QComboBox()
        combo.addItems(["<=", ">=", "="])
        combo.setCurrentText(op)
        self.tabela_restricoes.setCellWidget(linha, n, combo)
        self.tabela_restricoes.setCellWidget(linha, n + 1, _spinbox(b))

    def _remover_restricao(self) -> None:
        linha = self.tabela_restricoes.currentRow()
        if linha >= 0:
            self.tabela_restricoes.removeRow(linha)

    def _ler_objetivo(self) -> list[float]:
        valores = []
        for j in range(self.tabela_objetivo.columnCount()):
            w = self.tabela_objetivo.cellWidget(0, j)
            valores.append(w.value() if w is not None else 0.0)
        return valores

    def _ler_restricoes(self) -> list[dict]:
        restricoes = []
        n = self.tabela_restricoes.columnCount() - 2
        for i in range(self.tabela_restricoes.rowCount()):
            a = []
            for j in range(n):
                w = self.tabela_restricoes.cellWidget(i, j)
                a.append(w.value() if w is not None else 0.0)
            combo = self.tabela_restricoes.cellWidget(i, n)
            op = combo.currentText() if combo is not None else "<="
            wb = self.tabela_restricoes.cellWidget(i, n + 1)
            b = wb.value() if wb is not None else 0.0
            restricoes.append({"a": a, "op": op, "b": b})
        return restricoes

    def _atualizar_rotulo_z(self) -> None:
        c = self._ler_objetivo()
        termos = []
        for i, ci in enumerate(c, start=1):
            if ci != 0:
                termos.append(f"{ci:g}·x{i}")
        self.rotulo_z.setText("Z = " + (" + ".join(termos) if termos else "0"))

    # ---------------------------------------------------------------- exemplos
    def _carregar_exemplo_2(self) -> None:
        self._definir_problema(
            2,
            [3, 5],
            "max",
            [
                ([1, 0], "<=", 4),
                ([0, 2], "<=", 12),
                ([3, 2], "<=", 18),
            ],
        )

    def _carregar_exemplo_3(self) -> None:
        self._definir_problema(
            3,
            [2, 3, 4],
            "max",
            [
                ([1, 1, 1], "<=", 5),
                ([2, 0, 1], "<=", 7),
                ([0, 1, 2], "<=", 8),
            ],
        )

    def _definir_problema(self, n, c, tipo, restricoes) -> None:
        self.spin_n.blockSignals(True)
        self.spin_n.setValue(n)
        self.spin_n.blockSignals(False)
        self.combo_tipo.setCurrentText(tipo)
        self._n = n
        self._reconstruir_tabelas(list(c), [{"a": a, "op": op, "b": b} for a, op, b in restricoes])

    # ---------------------------------------------------------------- resolver
    def resolver(self) -> None:
        c = self._ler_objetivo()
        tipo = self.combo_tipo.currentText()
        restricoes = [(r["a"], r["op"], r["b"]) for r in self._ler_restricoes()]

        resultado = resolver_simplex(c, tipo, restricoes)
        self._passos = resultado.passos

        self.rotulo_status.setText(resultado.mensagem)
        cor = "#2e7d32"
        if resultado.status == STATUS_INVIAVEL:
            cor = "#c62828"
        elif resultado.status == STATUS_ILIMITADO:
            cor = "#e65100"
        self.rotulo_status.setStyleSheet(f"font-weight: bold; color: {cor};")

        self.timer.stop()
        self._reproduzindo = False
        self._atualizar_botao_reproduzir()
        if self._passos:
            self._mostrar(0)
        else:
            self.rotulo_passo.setText("")
            self.tabela_simplex.setRowCount(0)
            self.tabela_simplex.setColumnCount(0)

    # ------------------------------------------------------- navegação / animação
    def _mostrar(self, indice: int) -> None:
        if not self._passos:
            return
        self._indice = max(0, min(indice, len(self._passos) - 1))
        passo = self._passos[self._indice]
        self._renderizar_passos(passo)
        self._atualizar_estado_botoes()

    def proximo(self) -> None:
        self.timer.stop()
        self._reproduzindo = False
        self._atualizar_botao_reproduzir()
        if self._indice < len(self._passos) - 1:
            self._mostrar(self._indice + 1)

    def anterior(self) -> None:
        self.timer.stop()
        self._reproduzindo = False
        self._atualizar_botao_reproduzir()
        if self._indice > 0:
            self._mostrar(self._indice - 1)

    def pular_fim(self) -> None:
        self.timer.stop()
        self._reproduzindo = False
        self._atualizar_botao_reproduzir()
        self._mostrar(len(self._passos) - 1)

    def _alternar_reproducao(self) -> None:
        if self._reproduzindo:
            self.timer.stop()
            self._reproduzindo = False
        else:
            if self._indice >= len(self._passos) - 1:
                self._mostrar(0)
            self._reproduzindo = True
            self._definir_intervalo()
            self.timer.start()
        self._atualizar_botao_reproduzir()

    def _tick(self) -> None:
        if self._indice >= len(self._passos) - 1:
            self.timer.stop()
            self._reproduzindo = False
            self._atualizar_botao_reproduzir()
            return
        self._mostrar(self._indice + 1)

    def _definir_intervalo(self) -> None:
        self.timer.setInterval(max(120, 1500 - 150 * self.slider.value()))

    def _atualizar_botao_reproduzir(self) -> None:
        self.btn_reproduzir.setText("⏸ Pausar" if self._reproduzindo else "▶ Reproduzir")

    def _atualizar_estado_botoes(self) -> None:
        total = len(self._passos)
        self.btn_anterior.setEnabled(self._indice > 0)
        self.btn_proximo.setEnabled(self._indice < total - 1)

    # -------------------------------------------------------------- renderização
    def _renderizar_passos(self, passo: PassoSimplex) -> None:
        matriz = passo.matriz
        # matriz: (m+1) linhas x (n_vars + 1) colunas.
        n_vars = len(passo.colunas)
        m = matriz.shape[0] - 1  # nº de linhas de restrição

        self.tabela_simplex.clear()
        self.tabela_simplex.setRowCount(matriz.shape[0])
        self.tabela_simplex.setColumnCount(n_vars + 1)
        self.tabela_simplex.setHorizontalHeaderLabels(passo.colunas + ["b"])

        rotulos_linhas = list(passo.rotulos_linhas) + ["Z"]
        self.tabela_simplex.setVerticalHeaderLabels(rotulos_linhas)

        fonte_z = QFont()
        fonte_z.setBold(True)

        for r in range(matriz.shape[0]):
            for ccol in range(matriz.shape[1]):
                item = QTableWidgetItem(_fmt(float(matriz[r, ccol])))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if r == matriz.shape[0] - 1:
                    item.setFont(fonte_z)
                self.tabela_simplex.setItem(r, ccol, item)

        self._aplicar_destaques(passo, m, n_vars)

        # Texto do passo + contador.
        total = len(self._passos)
        self.rotulo_passo.setText(
            f"Passo {self._indice + 1} de {total} — {passo.texto}"
        )

        # Solução básica corrente (variáveis originais + Z).
        x = passo.solucao.get("x", {})
        z = passo.solucao.get("z")
        partes = [f"{k} = {_fmt(v)}" for k, v in x.items()]
        texto_sol = ", ".join(partes) if partes else "—"
        if z is not None:
            texto_sol += f"   |   Z = {_fmt(z)}"
        self.rotulo_solucao.setText(texto_sol)

    def _aplicar_destaques(self, passo: PassoSimplex, m: int, n_vars: int) -> None:
        col = passo.destaque_coluna
        linha = passo.destaque_linha
        pivo = passo.destaque_pivo

        fonte_negrito = QFont()
        fonte_negrito.setBold(True)

        if passo.otimo:
            # Tabela final: todas as células em verde suave.
            for r in range(self.tabela_simplex.rowCount()):
                for ccol in range(self.tabela_simplex.columnCount()):
                    item = self.tabela_simplex.item(r, ccol)
                    if item is not None:
                        item.setBackground(QColor(COR_OTIMO))
            self.rotulo_passo.setStyleSheet("color: #2e7d32; font-weight: bold;")
        else:
            self.rotulo_passo.setStyleSheet("")

            # Coluna pivô (variável que entra): roda a coluna inteira.
            if col is not None:
                for r in range(self.tabela_simplex.rowCount()):
                    item = self.tabela_simplex.item(r, col)
                    if item is not None:
                        item.setBackground(QColor(COR_COLUNA))

            # Linha pivô (variável que sai): apenas a linha de restrição.
            if linha is not None:
                for ccol in range(self.tabela_simplex.columnCount()):
                    item = self.tabela_simplex.item(linha, ccol)
                    if item is not None:
                        item.setBackground(QColor(COR_LINHA))

            # Elemento pivô (interseção): cor de contraste.
            if pivo is not None:
                pr, pc = pivo
                item = self.tabela_simplex.item(pr, pc)
                if item is not None:
                    item.setBackground(QColor(COR_PIVO))
                    item.setForeground(QColor("white"))
                    item.setFont(fonte_negrito)
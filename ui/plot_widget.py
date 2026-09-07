"""Widget do gráfico (matplotlib embutido) com animação passo a passo.

Responsável por desenhar o método gráfico em construção, na ordem:

1. eixos e grade;
2. cada reta de restrição (uma por vez) com seu rótulo;
3. região viável sombreada;
4. cada vértice (um por vez) com sua coordenada;
5. reta objetivo (linha de nível) deslizando até o vértice ótimo;
6. destaque do vértice ótimo e exibição do valor de Z.

A animação usa um `QTimer` que dispara `_proximo_passo()` a cada intervalo; cada
passo adiciona um elemento ao `Axes` e chama `canvas.draw_idle()`, exatamente o
padrão descrito em ARCHITECTURE.md. Os botões "Reproduzir" e "Pular" e o slider
de velocidade permitem controlar o ritmo (ou ver o resultado final direto).
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.geometria import TOL
from core.modelo import (
    OP_MAX,
    STATUS_ILIMITADO,
    STATUS_INVIAVEL,
    STATUS_MULTIPLAS,
    STATUS_OTIMO,
    FuncaoObjetivo,
    Restricao,
    Resultado,
)
from core.solver import resolver

# Paleta de cores usada para diferenciar as retas de restrição.
_PALETA = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _formatar(x: float) -> str:
    """Formata um float enxuto para exibição em rótulos."""
    return f"{x:.4g}"


def _estr_linear(a1: float, a2: float) -> str:
    """Constrói uma string legível tipo '3·x1 - 2·x2' tratando os sinais."""
    coefs = {"x1": a1, "x2": a2}
    termos: list[str] = []
    for var in ("x1", "x2"):
        coef = coefs[var]
        if abs(coef) < TOL:
            continue
        if abs(abs(coef) - 1.0) < TOL:
            termos.append(f"{'-' if coef < 0 else ''}{var}")
        else:
            termos.append(f"{coef:g}·{var}")

    if not termos:
        return "0"

    texto = termos[0]
    for t in termos[1:]:
        texto += (" + " if not t.startswith("-") else " - ") + t.lstrip("-")
    return texto


def _rotulo_restricao(r: Restricao, indice: int) -> str:
    """Rótulo exibido na reta, ex.: 'R1: x1 + 2·x2 <= 12'."""
    return f"R{indice}: {_estr_linear(r.a1, r.a2)} {r.operador} {_formatar(r.b)}"


def _pontos_reta(
    r: Restricao, xlim: tuple[float, float], ylim: tuple[float, float]
) -> tuple[list[float], list[float]]:
    """Devolve dois pontos (xs, ys) da reta, cobrindo toda a janela visível.

    Se a2 != 0, escreve x2 = (b - a1*x1)/a2 e varre x1; caso contrário a reta é
    vertical (x1 = b/a1) e varre x2.
    """
    if abs(r.a2) > TOL:
        x1s = [xlim[0], xlim[1]]
        x2s = [(r.b - r.a1 * x1) / r.a2 for x1 in x1s]
        return x1s, x2s
    if abs(r.a1) > TOL:
        xv = r.b / r.a1
        return [xv, xv], [ylim[0], ylim[1]]
    # Coeficientes nulos: reta degenerada; não há o que desenhar.
    return [], []


class PlotWidget(QWidget):
    """Canvas matplotlib + toolbar + controles de animação."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- Figura e eixos ---------------------------------------------------
        self.figure = Figure(figsize=(6.5, 5.2))
        self.figure.set_tight_layout(True)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # ---- Controles de animação -------------------------------------------
        self.btn_zoom_mais = QPushButton("Zoom +")
        self.btn_zoom_menos = QPushButton("Zoom −")
        self.btn_reproduzir = QPushButton("Reproduzir animação")
        self.btn_pular = QPushButton("Pular")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 10)
        self.slider.setValue(5)
        self.btn_zoom_mais.clicked.connect(lambda: self._zoom(2.0 / 3.0))
        self.btn_zoom_menos.clicked.connect(lambda: self._zoom(1.5))
        self.btn_reproduzir.clicked.connect(self._reproduzir)
        self.btn_pular.clicked.connect(self._pular)

        linha_controles = QHBoxLayout()
        linha_controles.addWidget(self.btn_zoom_mais)
        linha_controles.addWidget(self.btn_zoom_menos)
        linha_controles.addSpacing(16)
        linha_controles.addWidget(self.btn_reproduzir)
        linha_controles.addWidget(self.btn_pular)
        linha_controles.addWidget(QLabel("Lento"))
        linha_controles.addWidget(self.slider)
        linha_controles.addWidget(QLabel("Rápido"))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(linha_controles)

        # ---- Timer da animação ------------------------------------------------
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._proximo_passo)

        # ---- Estado atual -----------------------------------------------------
        self._passos: list = []       # lista de funções, uma por passo
        self._contador = 0            # quantos passos já foram executados
        self._resultado: Resultado | None = None
        self._objetivo: FuncaoObjetivo | None = None
        self._linha_objetivo = None  # referência da isoline para remover/atualizar
        self._texto_objetivo = None  # referência do texto "Z = k"

    # ------------------------------------------------------------------ utils
    def _zoom(self, fator: float) -> None:
        """Muda o zoom mantendo o centro do gráfico.

        `fator` < 1 aproxima (amplia) e `fator` > 1 afasta (diminui o zoom).
        """
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2.0
        cy = (ylim[0] + ylim[1]) / 2.0

        novo_xlim = (cx - (cx - xlim[0]) * fator, cx + (xlim[1] - cx) * fator)
        novo_ylim = (cy - (cy - ylim[0]) * fator, cy + (ylim[1] - cy) * fator)

        self.ax.set_xlim(novo_xlim)
        self.ax.set_ylim(novo_ylim)
        self.canvas.draw_idle()

    def _intervalo_ms(self) -> int:
        """Traduz o slider (1=lento .. 10=rápido) para milissegundos."""
        return max(80, 1500 - 150 * self.slider.value())

    def _obter_intervalo_timer(self) -> None:
        self.timer.setInterval(self._intervalo_ms())

    # ------------------------------------------------------ controle de animação
    def _reproduzir(self) -> None:
        """Reinicia a animação do zero — só se já houver um problema resolvido."""
        if not self._passos:
            return
        self.ax.clear()
        self._linha_objetivo = None
        self._texto_objetivo = None
        self._contador = 0
        self._obter_intervalo_timer()
        self.timer.start()

    def _pular(self) -> None:
        """Interrompe o timer e executa todos os passos restantes de uma vez."""
        if not self._passos:
            return
        self.timer.stop()
        while self._contador < len(self._passos):
            self._passos[self._contador]()
            self._contador += 1
        self.canvas.draw_idle()

    def _proximo_passo(self) -> None:
        """Executa o próximo passo da animação (chamado pelo QTimer)."""
        if self._contador >= len(self._passos):
            self.timer.stop()
            return
        self._passos[self._contador]()
        self._contador += 1
        self.canvas.draw_idle()

    # ---------------------------------------------------------- entrada do solver
    def resolver_problema(
        self,
        objetivo: FuncaoObjetivo,
        restricoes: list[Restricao],
        incluir_nao_negatividade: bool,
    ) -> Resultado:
        """Resolve o problema, monta os passos da animação e começa a desenhar."""
        self._objetivo = objetivo
        self._resultado = resolver(objetivo, restricoes, incluir_nao_negatividade)

        # Inclui também aqui as restrições ativas (para desenhar x1>=0 e x2>=0).
        self._restricoes_ativas = list(restricoes)
        if incluir_nao_negatividade:
            self._restricoes_ativas.append(Restricao(1.0, 0.0, ">=", 0.0))
            self._restricoes_ativas.append(Restricao(0.0, 1.0, ">=", 0.0))

        self._montar_passos()
        self._reproduzir()
        return self._resultado

    # ----------------------------------------------------------- montagem dos passos
    def _calcular_limites(self) -> tuple[float, float, float, float]:
        """Escolhe limites de eixo com base nos vértices + origem + margem."""
        xs = [0.0]
        ys = [0.0]
        for v in self._resultado.vertices:
            xs.append(v.x1)
            ys.append(v.x2)

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        if xmax - xmin < 1e-9:
            xmin, xmax = xmin - 5, xmax + 5
        if ymax - ymin < 1e-9:
            ymin, ymax = ymin - 5, ymax + 5

        pad_x = max(2.0, (xmax - xmin) * 0.30)
        pad_y = max(2.0, (ymax - ymin) * 0.30)
        return (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

    def _montar_passos(self) -> None:
        """Constrói a lista de funções (passos) que compõem a animação."""
        resultado = self._resultado
        verts = resultado.vertices
        xmin, xmax, ymin, ymax = self._calcular_limites()
        xlim = (xmin, xmax)
        ylim = (ymin, ymax)
        self._limites = (xlim, ylim)
        self._passos = []

        # Passo 0: eixos, grade e limites.
        self._passos.append(self._passo_eixos)

        # Passos 1..n: uma reta de restrição por passo.
        for i, r in enumerate(self._restricoes_ativas, start=1):
            self._passos.append(self._fazer_passo_restricao(r, i, xlim, ylim))

        # Passo seguinte: sombrear a região viável (se houver polígono).
        self._passos.append(self._passo_regiao)

        # Passos seguintes: um vértice por passo.
        # Centróide da região (fica dentro do polígono convexo): serve de
        # referência para afastar o rótulo de cada vértice PARA FORA da região,
        # evitando que as coordenadas se sobreponham.
        if verts:
            cx = sum(v.x1 for v in verts) / len(verts)
            cy = sum(v.x2 for v in verts) / len(verts)
        else:
            cx, cy = 0.0, 0.0
        for v in verts:
            self._passos.append(self._fazer_passo_vertice(v, cx, cy))

        # Passos da reta objetivo deslizando (se houver ótimo finito e objetivo não nulo).
        if (
            resultado.status in (STATUS_OTIMO, STATUS_MULTIPLAS)
            and resultado.z_otimo is not None
            and (abs(self._objetivo.c1) > TOL or abs(self._objetivo.c2) > TOL)
        ):
            for k in self._valores_k():
                self._passos.append(self._fazer_passo_isoline(k))

        # Passo final: destaque do ótimo + título/mensagem.
        self._passos.append(self._passo_final)

    # ---------------------------------------------------------------- passos
    def _passo_eixos(self) -> None:
        xlim, ylim = self._limites
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.grid(True, linestyle="--", alpha=0.5)
        # Eixos coordenados em destaque (linha x2=0 e x1=0).
        self.ax.axhline(0, color="black", linewidth=0.8)
        self.ax.axvline(0, color="black", linewidth=0.8)
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")

    def _fazer_passo_restricao(self, r, indice, xlim, ylim):
        cor = _PALETA[(indice - 1) % len(_PALETA)]

        def passo() -> None:
            xs, ys = _pontos_reta(r, xlim, ylim)
            if xs:
                self.ax.plot(xs, ys, color=cor, linewidth=2, label=_rotulo_restricao(r, indice))
                # Rotula a reta perto de uma das extremidades (20% ou 80%) para evitar o centro.
                fator = 0.2 if indice % 2 == 1 else 0.8
                xm = xs[0] + fator * (xs[1] - xs[0])
                ym = ys[0] + fator * (ys[1] - ys[0])
                self.ax.annotate(
                    _rotulo_restricao(r, indice),
                    xy=(xm, ym),
                    xytext=(4, 6),
                    textcoords="offset points",
                    color=cor,
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=cor, alpha=0.8, linewidth=0.6),
                )

        return passo

    def _passo_regiao(self) -> None:
        verts = self._resultado.vertices

        def passo() -> None:
            if len(verts) >= 3:
                xs = [v.x1 for v in verts]
                ys = [v.x2 for v in verts]
                self.ax.fill(xs, ys, alpha=0.25, color="green", label="Região viável")
            elif len(verts) == 2:
                # Região degenerada num segmento: desenha a aresta fechada.
                self.ax.plot(
                    [verts[0].x1, verts[1].x1],
                    [verts[0].x2, verts[1].x2],
                    color="green",
                    linewidth=2,
                )

        return passo

    def _fazer_passo_vertice(self, v, cx, cy):
        def passo() -> None:
            self.ax.scatter([v.x1], [v.x2], color="darkblue", s=40, zorder=5)

            # Direção do rótulo: afastando-se do centróide (para fora da região).
            dx = v.x1 - cx
            dy = v.x2 - cy
            norma = math.hypot(dx, dy)
            if norma < 1e-9:
                dx, dy = 1.0, 1.0  # vértice único/coincidente com o centróide
            else:
                dx, dy = dx / norma, dy / norma

            offset = 18.0  # distância maior em pontos para evitar sobreposição com o vértice e outras retas
            self.ax.annotate(
                f"({_formatar(v.x1)}, {_formatar(v.x2)})",
                xy=(v.x1, v.x2),
                xytext=(dx * offset, dy * offset),
                textcoords="offset points",
                fontsize=8,
                color="darkblue",
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkblue", alpha=0.8, linewidth=0.6),
            )

        return passo

    def _valores_k(self) -> list[float]:
        """Calcula a sequência de níveis k da isoline (do fora da região até Z*)."""
        verts = self._resultado.vertices
        if not verts:
            return []
        zs = [v.z for v in verts if v.z is not None]
        z_otimo = self._resultado.z_otimo
        zmin, zmax = min(zs), max(zs)
        amplitude = zmax - zmin if zmax - zmin > 1e-9 else 1.0

        if self._objetivo.tipo == OP_MAX:
            k_inicio = zmin - 0.25 * amplitude
        else:
            k_inicio = zmax + 0.25 * amplitude

        return list(np.linspace(k_inicio, z_otimo, 10))

    def _fazer_passo_isoline(self, k):
        def passo() -> None:
            # Remove a isoline anterior, se houver.
            if self._linha_objetivo is not None:
                self._linha_objetivo.remove()
                self._linha_objetivo = None
            if self._texto_objetivo is not None:
                self._texto_objetivo.remove()
                self._texto_objetivo = None

            xlim, ylim = self._limites
            c1, c2 = self._objetivo.c1, self._objetivo.c2
            if abs(c2) > TOL:
                x1s = [xlim[0], xlim[1]]
                x2s = [(k - c1 * x1) / c2 for x1 in x1s]
            elif abs(c1) > TOL:
                xv = k / c1
                x1s, x2s = [xv, xv], [ylim[0], ylim[1]]
            else:
                return

            self._linha_objetivo = self.ax.plot(
                x1s, x2s, color="crimson", linestyle="--", linewidth=1.6
            )[0]
            self._texto_objetivo = self.ax.text(
                0.02,
                0.95,
                f"Z = {_formatar(k)}",
                transform=self.ax.transAxes,
                color="crimson",
                fontsize=10,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=0.9, linewidth=0.8),
            )

        return passo

    def _passo_final(self) -> None:
        resultado = self._resultado

        def passo() -> None:
            otimos = resultado.vertices_otimos

            if resultado.status == STATUS_INVIAVEL:
                self.ax.set_title("Região inviável — não há solução", color="red")
            elif resultado.status == STATUS_ILIMITADO:
                self.ax.set_title("Problema ilimitado — sem ótimo finito", color="red")
            else:
                texto_tipo = "máx" if self._objetivo.tipo == OP_MAX else "mín"
                self.ax.set_title(
                    f"Z* (método gráfico) = {_formatar(resultado.z_otimo)}  ({texto_tipo})"
                )

            # Destaque dos vértices ótimos.
            for v in otimos:
                self.ax.scatter([v.x1], [v.x2], color="red", marker="*", s=220, zorder=6)

            # Em múltiplas soluções, destaca a aresta que une os vértices ótimos.
            if resultado.status == STATUS_MULTIPLAS and len(otimos) >= 2:
                p0, p1 = otimos[0], otimos[-1]
                self.ax.plot(
                    [p0.x1, p1.x1],
                    [p0.x2, p1.x2],
                    color="red",
                    linewidth=3,
                    label="Aresta ótima",
                )

        return passo
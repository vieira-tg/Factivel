"""Formatação de números para exibição (regra do projeto: NUNCA truncar).

Antes, cada widget tinha sua própria função de formatação com `f"{x:.4g}"`
(4 algarismos significativos), o que fazia valores grandes aparecerem em
notação científica e truncados:

    3000000        -> "3e+06"      (era exibido assim no gráfico/tableau)
    205882.35...   -> "2.059e+05"
    14000          -> "1.4e+04"

A função única `formatar_numero` abaixo resolve isso: devolve sempre um
decimal limpo, com todos os dígitos, sem notação científica, removendo apenas
o ruído de ponto flutuante (ex.: 0.5800000000000001 -> "0.58").

Detalhe importante: NUNCA usamos `round(x)` desprotegido nem formatadores
com casas fixas (`.4f` / `.6f` / `.4g`), justamente para não perder dígitos
de valores grandes nem apresentar zeros desnecessários.
"""

from __future__ import annotations

from decimal import Decimal

# Número de algarismos significativos mantidos. 12 é mais que suficiente para
# esconder o ruído típico de ponto flutuante binário (que começa lá pelo 15º
# dígito) sem cortar nenhum dígito relevante do problema.
_ALGARISMOS = 12


def formatar_numero(valor: float) -> str:
    """Formata um float como decimal limpo, sem truncar e sem notação científica.

    Algoritmo (passo a passo, didático):
    1. Converte para string de 12 algarismos significativos — isso elimina o
       "lixo" do float (2.7999999999999998 vira 2.8) sem perder dígitos reais.
    2. Passa por `Decimal` e formata com `f` — assim o texto NUNCA usa
       notação científica (e+06), não importa quão grande ou pequeno seja.
    3. Remove zeros à direita depois da vírgula (12.000 -> "12",
       0.580 -> "0.58").
    """
    # "g" com 12 algarismos significativos: captura o valor real e descarta os
    # últimos dígitos de ruído de ponto flutuante.
    texto = f"{valor:.{_ALGARISMOS}g}"

    # Decimal + formato "f": expansão despejada em decimal puro, por extenso,
    # então "3e+06" vira "3000000" e "1e-08" vira "0.00000001".
    texto = format(Decimal(texto), "f")

    # Remove zeros excedentes à direita (só na parte fracionária).
    if "." in texto:
        texto = texto.rstrip("0").rstrip(".")

    return texto

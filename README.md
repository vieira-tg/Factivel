# Factível

**Factível** é um aplicativo desktop desenvolvido em Python para resolver problemas de **Programação Linear (PL)** de forma intuitiva, visual e didática.

O software suporta dois métodos principais:
1. **Método Gráfico (2 variáveis: $x_1, x_2$)**: Desenha as retas de restrição, sombreia a região viável, calcula e destaca os vértices, avalia a função objetivo e apresenta uma **animação passo a passo** da construção do gráfico (eixos $\rightarrow$ retas $\rightarrow$ região $\rightarrow$ vértices $\rightarrow$ linha de nível deslizando até o ótimo).
2. **Método Simplex ($N$ variáveis)**: Resolve problemas com $N$ variáveis utilizando o método de **Duas Fases**, exibindo o tableau animado a cada pivô com navegação manual e automática.

---

## 🚀 Funcionalidades

- **Visualização Animada**: Acompanhe o desenho do gráfico passo a passo com controles de reprodução, pausa, salto e velocidade.
- **Casos Especiais**: Tratamento explícito e amigável para regiões **inviáveis**, **ilimitadas** e com **múltiplas soluções ótimas** (destacando a aresta ótima).
- **Validação Cruzada**: Todos os cálculos são validados contra o `scipy.optimize.linprog` como segunda fonte de verdade.
- **Exportação & Persistência**: Exporte o gráfico gerado como imagem ou salve/carregue seus problemas em arquivos JSON.
- **Interface Moderna**: Desenvolvido com **PySide6** e **Matplotlib**, com identidade visual própria e ícones customizados.

---

## 🛠️ Requisitos

- **Python 3.11** ou superior.
- Bibliotecas dependentes (conforme `requirements.txt`):
  - `PySide6 >= 6.6`
  - `matplotlib >= 3.8`
  - `numpy >= 1.26`
  - `scipy >= 1.11`
  - `pytest >= 8.0` (para testes)

---

## ⚙️ Como Instalar e Rodar

1. **Abra o terminal** na pasta raiz do projeto (`D:\ProgramacaoLinear`).

2. **Crie e ative um ambiente virtual** (opcional, mas recomendado):
   ```bash
   python -m venv .venv
   
   # No Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   
   # No Windows (CMD):
   .venv\Scripts\activate.bat
   ```

3. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute o aplicativo**:
   Entre na pasta `Factivel` e execute o script principal:
   ```bash
   cd Factivel
   python main.py
   ```

---

## 🧪 Executando os Testes

O projeto possui uma suíte robusta de testes unitários cobrindo o solver gráfico, o algoritmo Simplex, degenerescência e validações cruzadas com o SciPy.

Para rodar os testes:
```bash
# A partir da pasta Factivel (ou configurando o pythonpath)
cd Factivel
pytest
```

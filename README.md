# Try-on Virtual de Maquiagem 💄

Experimente batom, blush e sombra virtualmente usando a webcam! O programa
detecta seu rosto em tempo real com o **MediaPipe Face Mesh**, encontra os
lábios, as maçãs do rosto e as pálpebras e aplica a maquiagem de forma
realista, acompanhando os movimentos da cabeça. Dá para combinar livremente
batom, blush e sombra, escolhendo as cores com o mouse num painel ao lado
do vídeo.

Projeto de estudo feito em Python com OpenCV, MediaPipe e NumPy.

## Como funciona

1. O **OpenCV** abre a webcam e lê cada quadro do vídeo.
2. O **MediaPipe** encontra 478 pontos no rosto. Usamos os pontos do
   contorno dos lábios, das maçãs do rosto, dos olhos e das sobrancelhas.
3. Com esses pontos criamos **máscaras**:
   - **batom:** a área dos lábios, sem o "buraco" da boca;
   - **blush:** várias camadas de ovais desfocados, somadas, na maçã do
     rosto, começando embaixo do olho e subindo na diagonal até a têmpora.
     O blush acompanha o tamanho e a inclinação do rosto;
   - **sombra:** a pálpebra, entre a linha dos cílios e a sobrancelha,
     sem a parte aberta do olho. Como é refeita a cada quadro, acompanha
     a pálpebra quando a pessoa pisca.
4. As máscaras são **desfocadas** (GaussianBlur), para não ter borda marcada.
5. A cor é misturada no **espaço de cor LAB**: os canais de cor (A e B)
   se aproximam da cor da maquiagem, e o brilho (canal L) é mantido, então
   a luz e a textura da pele continuam. Assim a pele não fica "chapada".
6. Toques de realismo:
   - **batom:** cor menos saturada, degradê (bordas um pouco mais escuras,
     centro mais claro) e um **gloss** sutil no meio dos lábios;
   - **blush:** pigmento forte no centro que some aos poucos, e um
     **iluminador** leve no topo da maçã do rosto;
   - **sombra:** duas camadas de degradê somadas, fortes perto dos cílios
     e sumindo em direção à sobrancelha; o champanhe tem um brilho sutil.

   Os batons cereja escuro, vinho/berry e marrom-chocolate usam o efeito
   original (cor por igual, sem gloss).
7. Para ficar rápido, a maquiagem é calculada só num recorte em volta da
   boca, das bochechas e dos olhos, e os desfoques grandes são feitos numa imagem menor.
8. Um **painel** desenhado com o OpenCV fica ao lado do vídeo, e o
   `cv2.setMouseCallback` detecta os cliques nas abas e nas bolinhas de cor.

## Como instalar

É preciso ter o **Python 3.9 ou mais novo** e uma webcam.

```bash
# 1. Clone o repositório e entre na pasta
git clone https://github.com/marianymota/maquiagem-ia.git
cd maquiagem-ia

# 2. (Recomendado) Crie e ative um ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt
```

## Como rodar

```bash
python main.py
```

Na primeira execução, o programa baixa automaticamente o modelo do
MediaPipe (`face_landmarker.task`, cerca de 4 MB).

### Controles

Batom, blush e sombra são independentes: escolha um de cada e combine.

**Com o mouse:** no painel ao lado do vídeo, clique na aba **Batom**,
**Blush** ou **Sombra** e depois na bolinha da cor. A cor escolhida fica
com contorno branco. A bolinha com **X** tira a maquiagem daquela aba.

**Com o teclado (atalhos):**

| Tecla | Batom              |
|-------|--------------------|
| `1`   | Nude rosado        |
| `2`   | Rosa-chá           |
| `3`   | Coral              |
| `4`   | Vermelho clássico  |
| `5`   | Cereja escuro      |
| `6`   | Vinho/berry        |
| `7`   | Marrom-chocolate   |
| `8`   | Malva              |
| `0`   | Tirar o batom      |

| Tecla | Blush              |
|-------|--------------------|
| `a`   | Rosa-pêssego claro |
| `s`   | Rosé               |
| `d`   | Coral vibrante     |
| `f`   | Terracota          |
| `g`   | Malva              |
| `h`   | Berry              |
| `b`   | Liga/desliga blush |

| Tecla | Sombra                 |
|-------|------------------------|
| `z`   | Rosé                   |
| `x`   | Pêssego                |
| `c`   | Bronze/cobre           |
| `v`   | Marrom suave           |
| `n`   | Malva                  |
| `m`   | Champanhe (com brilho) |
| `j`   | Ameixa                 |
| `o`   | Liga/desliga sombra    |

| Tecla | Ação               |
|-------|--------------------|
| `q`   | Sair do programa   |

A combinação atual (batom + blush + sombra) aparece escrita no painel.

As cores ficam nos dicionários `BATONS`, `BLUSHES` e `SOMBRAS` no topo do
`main.py`, no formato `tecla: (nome, cor BGR, opacidade)` (a sombra tem um
quarto valor, `True` para as cores com brilho), para facilitar a troca.

> Dica: clique na janela do vídeo antes de apertar as teclas.

## Próximos passos

- [x] Mais cores de batom e blush, com combinação livre
- [x] Painel clicável para escolher as cores com o mouse
- [x] Sombra nos olhos, com aba própria no painel
- [ ] Controle de intensidade da maquiagem pelo teclado
- [ ] Aplicar outras maquiagens: delineado, rímel e iluminador separado
- [ ] Botão para tirar uma foto (salvar a imagem com a maquiagem)
- [ ] Suavizar o movimento dos pontos para o batom tremer menos
- [ ] Suportar mais de um rosto ao mesmo tempo

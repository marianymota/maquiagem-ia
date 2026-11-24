# Try-on Virtual de Maquiagem 💄

Experimente batom e blush virtualmente usando a webcam! O programa detecta
seu rosto em tempo real com o **MediaPipe Face Mesh**, encontra os lábios e
as maçãs do rosto e aplica a maquiagem de forma realista, acompanhando os
movimentos da cabeça. Dá para combinar qualquer batom com qualquer blush,
escolhendo as cores com o mouse num painel ao lado do vídeo.

Projeto de estudo feito em Python com OpenCV, MediaPipe e NumPy.

## Como funciona

1. O **OpenCV** abre a webcam e lê cada quadro do vídeo.
2. O **MediaPipe** encontra 478 pontos no rosto. Usamos os pontos do
   contorno dos lábios e os pontos das maçãs do rosto.
3. Com esses pontos criamos **máscaras**:
   - **batom:** a área dos lábios, sem o "buraco" da boca;
   - **blush:** várias camadas de ovais desfocados, somadas, na maçã do
     rosto, começando embaixo do olho e subindo na diagonal até a têmpora.
     O blush acompanha o tamanho e a inclinação do rosto.
4. As máscaras são **desfocadas** (GaussianBlur), para não ter borda marcada.
5. A cor é misturada no **espaço de cor LAB**: os canais de cor (A e B)
   se aproximam da cor da maquiagem, e o brilho (canal L) é mantido, então
   a luz e a textura da pele continuam. Assim a pele não fica "chapada".
6. Toques de realismo:
   - **batom:** cor menos saturada, degradê (bordas um pouco mais escuras,
     centro mais claro) e um **gloss** sutil no meio dos lábios;
   - **blush:** pigmento forte no centro que some aos poucos, e um
     **iluminador** leve no topo da maçã do rosto.

   Os batons cereja escuro, vinho/berry e marrom-chocolate usam o efeito
   original (cor por igual, sem gloss).
7. Para ficar rápido, a maquiagem é calculada só num recorte em volta da
   boca e das bochechas, e os desfoques grandes são feitos numa imagem menor.
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

O batom e o blush são independentes: escolha um de cada e combine.

**Com o mouse:** no painel ao lado do vídeo, clique na aba **Batom** ou
**Blush** e depois na bolinha da cor. A cor escolhida fica com contorno
branco. A bolinha com **X** tira o batom ou o blush.

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

| Tecla | Ação               |
|-------|--------------------|
| `q`   | Sair do programa   |

A combinação atual (batom + blush) aparece escrita no painel.

As cores ficam nos dicionários `BATONS` e `BLUSHES` no topo do `main.py`,
no formato `tecla: (nome, cor BGR, opacidade)`, para facilitar a troca.

> Dica: clique na janela do vídeo antes de apertar as teclas.

## Próximos passos

- [x] Mais cores de batom e blush, com combinação livre
- [x] Painel clicável para escolher as cores com o mouse
- [ ] Controle de intensidade da maquiagem pelo teclado
- [ ] Aplicar outras maquiagens: sombra, delineado e iluminador
- [ ] Botão para tirar uma foto (salvar a imagem com a maquiagem)
- [ ] Suavizar o movimento dos pontos para o batom tremer menos
- [ ] Suportar mais de um rosto ao mesmo tempo

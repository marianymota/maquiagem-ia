# Try-on Virtual de Maquiagem 💄

Experimente batom e blush virtualmente usando a webcam! O programa detecta
seu rosto em tempo real com o **MediaPipe Face Mesh**, encontra os lábios e
as maçãs do rosto e aplica a maquiagem de forma realista, acompanhando os
movimentos da cabeça. Dá para combinar qualquer batom com qualquer blush.

Projeto de estudo feito em Python com OpenCV, MediaPipe e NumPy.

## Como funciona

1. O **OpenCV** abre a webcam e lê cada quadro do vídeo.
2. O **MediaPipe** encontra 478 pontos no rosto. Usamos os pontos do
   contorno dos lábios e os pontos das maçãs do rosto.
3. Com esses pontos criamos **máscaras**:
   - **batom:** a área dos lábios, sem o "buraco" da boca;
   - **blush:** dois ovais que acompanham o tamanho e a inclinação do rosto.
4. As máscaras são **desfocadas** (GaussianBlur), para não ter borda marcada.
5. A cor é misturada no **espaço de cor LAB**: o canal L (brilho, luz e
   textura da pele) é preservado, e só os canais de cor (A e B) mudam,
   com opacidade baixa (20% a 35%). Assim a pele não fica "chapada".

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

| Tecla | Batom              |
|-------|--------------------|
| `1`   | Nude rosado        |
| `2`   | Rosa-chá           |
| `3`   | Coral              |
| `4`   | Vermelho clássico  |
| `5`   | Cereja             |
| `6`   | Vinho/berry        |
| `7`   | Marrom-chocolate   |
| `8`   | Malva              |
| `0`   | Tirar o batom      |

| Tecla | Blush              |
|-------|--------------------|
| `a`   | Rosa-pêssego       |
| `s`   | Rosé               |
| `d`   | Coral suave        |
| `f`   | Terracota          |
| `g`   | Malva              |
| `h`   | Berry suave        |
| `b`   | Liga/desliga blush |

| Tecla | Ação               |
|-------|--------------------|
| `q`   | Sair do programa   |

A combinação atual (batom + blush) aparece no canto da tela.

As cores ficam nos dicionários `BATONS` e `BLUSHES` no topo do `main.py`,
no formato `tecla: (nome, cor BGR, opacidade)`, para facilitar a troca.

> Dica: clique na janela do vídeo antes de apertar as teclas.

## Próximos passos

- [x] Mais cores de batom e blush, com combinação livre
- [ ] Controle de intensidade da maquiagem pelo teclado
- [ ] Aplicar outras maquiagens: sombra, delineado e iluminador
- [ ] Botão para tirar uma foto (salvar a imagem com a maquiagem)
- [ ] Suavizar o movimento dos pontos para o batom tremer menos
- [ ] Suportar mais de um rosto ao mesmo tempo
- [ ] Criar uma interface gráfica para escolher as cores

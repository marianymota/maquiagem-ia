# Try-on Virtual de Maquiagem 💄

Experimente batom virtualmente usando a webcam! O programa detecta seu rosto
em tempo real com o **MediaPipe Face Mesh**, encontra o contorno dos lábios
e pinta essa área com uma cor semitransparente, como se fosse um batom.

Projeto de estudo feito em Python com OpenCV, MediaPipe e NumPy.

## Como funciona

1. O **OpenCV** abre a webcam e lê cada quadro do vídeo.
2. O **MediaPipe** encontra 478 pontos no rosto; usamos os pontos que
   formam o contorno dos lábios.
3. Com esses pontos criamos uma **máscara** (a área dos lábios, sem o
   "buraco" da boca) e misturamos a cor do batom com a imagem original
   usando **NumPy**.

## Como instalar

É preciso ter o **Python 3.9 ou mais novo** e uma webcam.

```bash
# 1. Clone o repositório e entre na pasta
git clone <url-do-repositorio>
cd tryon-maquiagem

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

| Tecla | Ação              |
|-------|-------------------|
| `1`   | Nude rosado       |
| `2`   | Rosa suave        |
| `3`   | Vermelho clássico |
| `4`   | Vinho             |
| `5`   | Coral             |
| `q`   | Sair do programa  |

As cores, a opacidade e a suavização da borda ficam no topo do `main.py`
(dicionário `CORES` e constantes `OPACIDADE` e `SUAVIZAR_BORDA`), para
facilitar a troca.

> Dica: clique na janela do vídeo antes de apertar as teclas.

## Próximos passos

- [ ] Adicionar mais cores e um controle de intensidade do batom
- [ ] Aplicar outras maquiagens: blush, sombra e delineado
- [ ] Botão para tirar uma foto (salvar a imagem com a maquiagem)
- [ ] Suavizar o movimento dos pontos para o batom tremer menos
- [ ] Suportar mais de um rosto ao mesmo tempo
- [ ] Criar uma interface gráfica para escolher as cores

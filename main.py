# Try-on virtual de maquiagem com a webcam (batom + blush)
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# Arquivo do modelo de pontos do rosto (Face Mesh) do MediaPipe.
# Se ele não existir na pasta, o programa baixa automaticamente.
MODELO = "face_landmarker.task"
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# ======================= CONFIGURAÇÕES DE COR =======================
# Atenção: o OpenCV usa a ordem BGR (azul, verde, vermelho), não RGB!
# (os nomes ficam sem acento porque o texto do OpenCV não mostra acentos)
#
# Batons: tecla -> (nome, cor BGR, opacidade)
# A opacidade diz quanto a cor da boca se aproxima da cor do batom
# (0 = nada, 1 = totalmente). O brilho e a textura continuam da boca.
BATONS = {
    "1": ("Nude rosado", (130, 140, 200), 0.32),
    "2": ("Rosa-cha", (125, 110, 190), 0.36),
    "3": ("Coral", (70, 95, 240), 0.36),
    "4": ("Vermelho classico", (30, 20, 200), 0.42),
    "5": ("Cereja escuro", (40, 10, 130), 0.50),
    "6": ("Vinho/berry", (70, 25, 105), 0.50),
    "7": ("Marrom-chocolate", (35, 50, 95), 0.50),
    "8": ("Malva", (140, 95, 165), 0.36),
}

# Batons que usam o efeito ORIGINAL (cor por igual, sem degradê nem gloss).
# Eles já estão do jeito certo, então não mudam.
BATONS_ORIGINAIS = {"5", "6", "7"}

# Efeito natural (usado pelos outros batons):
SATURACAO_BATOM = 0.8  # 1 = cor original; menor = cor mais "apagada"
DEGRADE_BATOM = 10     # quanto as bordas escurecem e o centro clareia
GLOSS_BATOM = 0.30     # força do brilho (gloss) no centro dos lábios

# Blushes: tecla -> (nome, cor BGR, opacidade)
BLUSHES = {
    "a": ("Rosa-pessego claro", (150, 160, 250), 0.40),
    "s": ("Rose", (150, 120, 225), 0.35),
    "d": ("Coral vibrante", (80, 110, 255), 0.35),
    "f": ("Terracota", (55, 75, 185), 0.40),
    "g": ("Malva", (165, 110, 180), 0.35),
    "h": ("Berry", (100, 50, 170), 0.35),
}

# Quanto a maquiagem pode escurecer/clarear a pele (0 = só muda a cor,
# 1 = brilho igual ao da cor). Todos os pixels mudam igual, então a
# textura continua. Ajuda tons escuros (vinho, chocolate) a aparecerem.
# Sombras: tecla -> (nome, cor BGR, opacidade, tem_brilho)
# A opacidade é a do ponto mais forte (perto dos cílios); ela vai
# diminuindo até sumir perto da sobrancelha.
SOMBRAS = {
    "z": ("Rose", (140, 120, 210), 0.55, False),
    "x": ("Pessego", (110, 155, 240), 0.55, False),
    "c": ("Bronze/cobre", (45, 90, 175), 0.55, False),
    "v": ("Marrom suave", (65, 85, 125), 0.55, False),
    "n": ("Malva", (155, 100, 165), 0.55, False),
    "m": ("Champanhe (brilho)", (160, 200, 230), 0.50, True),
    "j": ("Ameixa", (80, 35, 105), 0.55, False),
}
BRILHO_SOMBRA = 0.5    # quanto a sombra escurece/clareia a pálpebra
SHIMMER = 0.12         # força do brilho das sombras com tem_brilho=True

BRILHO_BATOM = 0.8     # usado só pelos batons 5, 6 e 7
BRILHO_BLUSH = 0.10
ILUMINADOR = 0.10      # força do brilho no topo da maçã do rosto
# ====================================================================

# O Face Mesh encontra 478 pontos no rosto. Estes são os números (índices)
# dos pontos que formam o contorno dos lábios, já em ordem, dando a volta.
LABIOS_EXTERNO = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                  291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
# Contorno de dentro da boca (para não pintar os dentes com a boca aberta)
LABIOS_INTERNO = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                  308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# Pontos usados para o blush, um grupo para cada lado do rosto:
# - "inicio": pontos cuja média é onde o blush começa (maçã, embaixo do olho)
# - "tempora": ponto para onde o blush sobe e se alonga (como na maquiagem real)
# - "luz": pontos do topo da maçã do rosto, onde vai o iluminador
BLUSH_LADOS = [
    {"inicio": [50, 123], "tempora": 139, "luz": [116, 117]},
    {"inicio": [280, 352], "tempora": 368, "luz": [345, 346]},
]

# Camadas do blush: (tamanho, peso). Cada camada é um oval desfocado;
# somadas, ficam fortes no centro e somem devagar nas pontas.
CAMADAS_BLUSH = [(1.0, 0.35), (0.7, 0.35), (0.4, 0.30)]
# Pontos usados para a sombra, um grupo para cada olho:
# - "olho": contorno da parte aberta do olho, em ordem (não pode ser pintada)
# - "palpebra": linha dos cílios de cima, do canto de fora ao de dentro
# - "sobrancelha": parte de baixo da sobrancelha, de dentro para fora
# - "topo_olho" e "meio_sobrancelha": para medir a altura da pálpebra
OLHOS = [
    {"olho": [33, 246, 161, 160, 159, 158, 157, 173, 133,
              155, 154, 153, 145, 144, 163, 7],
     "palpebra": [33, 246, 161, 160, 159, 158, 157, 173, 133],
     "sobrancelha": [55, 65, 52, 53, 46],
     "topo_olho": 159, "meio_sobrancelha": 52},
    {"olho": [263, 466, 388, 387, 386, 385, 384, 398, 362,
              382, 381, 380, 374, 373, 390, 249],
     "palpebra": [263, 466, 388, 387, 386, 385, 384, 398, 362],
     "sobrancelha": [285, 295, 282, 283, 276],
     "topo_olho": 386, "meio_sobrancelha": 282},
]

LATERAL_ESQUERDA = 234  # bordas do rosto, para medir a largura
LATERAL_DIREITA = 454

# ========================= PAINEL CLICÁVEL ==========================
JANELA = "Try-on de Maquiagem"
PAINEL_LARGURA = 270        # largura do painel ao lado do vídeo
ABA_ALTURA = 60             # altura dos botões das abas
RAIO_BOLINHA = 22           # tamanho das bolinhas de cor
COLUNAS_X = [50, 135, 220]  # posição x das 3 colunas de bolinhas
LINHA_Y0 = 100              # posição y da primeira linha de bolinhas
ESPACO_Y = 65               # distância entre as linhas

COR_FUNDO = (45, 40, 40)
COR_ABA_ATIVA = (90, 80, 80)
COR_ABA_INATIVA = (55, 50, 50)
BRANCO = (255, 255, 255)
CINZA = (160, 160, 160)
# ====================================================================


# ---------------------------- DETECÇÃO -----------------------------

def criar_detector():
    """Baixa o modelo (se precisar) e cria o detector de pontos do rosto."""
    if not os.path.exists(MODELO):
        print("Baixando o modelo do MediaPipe...")
        urllib.request.urlretrieve(URL_MODELO, MODELO)

    opcoes = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELO),
        num_faces=1,  # procura só um rosto
    )
    return vision.FaceLandmarker.create_from_options(opcoes)


def detectar_rosto(detector, frame):
    """Detecta o rosto e devolve os 478 pontos em pixels (array N x 2).

    Retorna None se nenhum rosto for encontrado.
    """
    # O OpenCV usa cores em BGR, mas o MediaPipe espera RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    resultado = detector.detect(imagem)

    if not resultado.face_landmarks:
        return None

    rosto = resultado.face_landmarks[0]
    altura, largura = frame.shape[:2]

    # Os pontos vêm "normalizados" (de 0 a 1); multiplicamos pelo
    # tamanho da imagem para converter em pixels
    return np.array([(p.x * largura, p.y * altura) for p in rosto], dtype=np.float32)


# ---------------------------- MAQUIAGEM ----------------------------

def aplicar_cor(frame, peso, cor, brilho=0.0):
    """Mistura uma cor na imagem de forma realista, usando o espaço LAB.

    No LAB, o canal L é o brilho (luz e sombra, textura da pele) e os
    canais A e B são só a cor. Mexendo principalmente em A e B, a cor muda,
    mas as rugas, o brilho e as sombras da pele continuam lá.

    peso: imagem 2D com valores de 0 a 1 (onde e quanto aplicar a cor)
    cor: cor BGR da maquiagem
    brilho: 0 = não mexe no brilho; maior = aproxima o brilho do da cor
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Converte a cor da maquiagem para LAB também
    cor_lab = cv2.cvtColor(np.uint8([[cor]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    L, A, B = cv2.split(lab)

    # Aproxima a cor (A e B) da cor da maquiagem, na proporção do peso
    A = A * (1 - peso) + cor_lab[1] * peso
    B = B * (1 - peso) + cor_lab[2] * peso

    # Opcional: desloca o brilho de toda a área pela mesma quantidade.
    # Como todos os pixels mudam igual, a textura (diferenças) se mantém.
    if brilho > 0 and np.any(peso > 0.05):
        brilho_medio = L[peso > 0.05].mean()
        L = L + (cor_lab[0] - brilho_medio) * peso * brilho

    lab = cv2.merge([L, A, B])
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def passar_batom(frame, pontos, cor, opacidade):
    """Pinta os lábios com a cor do batom."""
    externo = pontos[LABIOS_EXTERNO].astype(np.int32)
    interno = pontos[LABIOS_INTERNO].astype(np.int32)

    # 1) Máscara: imagem preta onde a área dos lábios fica branca
    mascara = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mascara, [externo], 255)  # preenche o contorno externo
    cv2.fillPoly(mascara, [interno], 0)    # "apaga" o buraco da boca

    # 2) Desfoca a borda para o batom não ficar "recortado".
    #    O tamanho do desfoque acompanha o tamanho da boca (precisa ser ímpar).
    largura_boca = np.linalg.norm(pontos[61] - pontos[291])
    k = int(largura_boca * 0.06) // 2 * 2 + 1
    mascara = cv2.GaussianBlur(mascara, (k, k), 0)

    # 3) Converte para pesos de 0 a 1, multiplicados pela opacidade
    peso = mascara.astype(np.float32) / 255.0 * opacidade
    return aplicar_cor(frame, peso, cor, brilho=BRILHO_BATOM)


def impar(valor):
    """Arredonda para um número ímpar (o GaussianBlur exige tamanho ímpar)."""
    return max(1, int(valor) // 2 * 2 + 1)


def clarear(L, mascara, forca):
    """Clareia o brilho (canal L) onde a máscara é forte, como um reflexo de luz.

    Usa (255 - L): partes escuras clareiam mais que as já claras, então
    as linhas e a textura continuam aparecendo por baixo do brilho.
    """
    return L + (255 - L) * mascara * forca


def oval_desfocado(formato, centro, eixos, angulo, desfoque, reducao=1):
    """Desenha um oval branco numa imagem preta e desfoca. Devolve 0 a 1.

    reducao: faz o desenho numa imagem N vezes menor e depois amplia.
    Como o oval é bem borrado, o resultado fica igual e MUITO mais rápido
    (desfocar uma imagem grande com um desfoque enorme é lento).
    """
    altura, largura = formato
    pequena = (altura // reducao, largura // reducao)
    img = np.zeros(pequena, dtype=np.uint8)
    cv2.ellipse(img, (int(centro[0] / reducao), int(centro[1] / reducao)),
                (max(1, int(eixos[0] / reducao)), max(1, int(eixos[1] / reducao))),
                angulo, 0, 360, 255, -1)
    k = impar(desfoque / reducao)
    img = cv2.GaussianBlur(img, (k, k), 0)
    if reducao > 1:
        img = cv2.resize(img, (largura, altura), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 255.0


def no_recorte(frame, pontos, indices, margem, funcao, *args):
    """Roda a função só num pedaço da imagem em volta de alguns pontos.

    Processar a imagem inteira é lento; a maquiagem só ocupa uma parte
    do rosto, então recortamos essa parte, aplicamos e colamos de volta.
    """
    altura, largura = frame.shape[:2]
    x, y, w, h = cv2.boundingRect(pontos[indices].astype(np.int32))
    margem = int(margem)
    x0, y0 = max(0, x - margem), max(0, y - margem)
    x1, y1 = min(largura, x + w + margem), min(altura, y + h + margem)
    if x1 <= x0 or y1 <= y0:
        return frame  # rosto fora da imagem

    resultado = frame.copy()
    recorte = np.ascontiguousarray(frame[y0:y1, x0:x1])
    # Os pontos passam a contar a partir do canto do recorte
    resultado[y0:y1, x0:x1] = funcao(recorte, pontos - np.float32([x0, y0]), *args)
    return resultado


def passar_batom_natural(frame, pontos, cor, opacidade):
    """Batom natural, calculado só na região da boca (mais rápido)."""
    largura_boca = np.linalg.norm(pontos[61] - pontos[291])
    return no_recorte(frame, pontos, LABIOS_EXTERNO, largura_boca * 0.3,
                      pintar_batom_natural, cor, opacidade)


def pintar_batom_natural(frame, pontos, cor, opacidade):
    """Batom com cara de real: cor misturada, degradê e um gloss sutil."""
    externo = pontos[LABIOS_EXTERNO].astype(np.int32)
    interno = pontos[LABIOS_INTERNO].astype(np.int32)
    largura_boca = np.linalg.norm(pontos[61] - pontos[291])

    # 1) Máscara dos lábios (sem o buraco da boca), com a borda suave
    labios = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(labios, [externo], 255)
    cv2.fillPoly(labios, [interno], 0)
    k = impar(largura_boca * 0.06)
    mascara = cv2.GaussianBlur(labios, (k, k), 0).astype(np.float32) / 255.0

    # 2) "Profundidade": 0 na borda dos lábios e 1 no meio deles.
    #    distanceTransform mede a distância de cada pixel até a borda.
    distancia = cv2.distanceTransform(labios, cv2.DIST_L2, 5)
    profundidade = np.clip(distancia / (largura_boca * 0.07), 0, 1)
    profundidade = cv2.GaussianBlur(profundidade, (k, k), 0)

    # 3) Passa para o LAB: L = brilho, A e B = cor
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    cor_lab = cv2.cvtColor(np.uint8([[cor]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    # 4) Cor do batom um pouco menos saturada (128 = sem cor no LAB)
    alvo_a = 128 + (cor_lab[1] - 128) * SATURACAO_BATOM
    alvo_b = 128 + (cor_lab[2] - 128) * SATURACAO_BATOM

    # 5) Mistura só a cor (A e B); o brilho L original fica, com a textura
    peso = mascara * opacidade
    A = A * (1 - peso) + alvo_a * peso
    B = B * (1 - peso) + alvo_b * peso

    # 6) Degradê: bordas levemente mais escuras, centro levemente mais claro
    ajuste = (profundidade - 0.5) * 2 * DEGRADE_BATOM  # vai de -DEGRADE a +DEGRADE
    L = L + ajuste * mascara

    # 7) Gloss: dois ovais de luz bem desfocados, só dentro dos lábios
    angulo = np.degrees(np.arctan2(*(pontos[291] - pontos[61])[::-1]))
    centro_inferior = pontos[14] + (pontos[17] - pontos[14]) * 0.45  # meio do lábio de baixo
    centro_superior = pontos[0] + (pontos[13] - pontos[0]) * 0.55    # meio do lábio de cima
    desfoque = largura_boca * 0.10
    gloss = (oval_desfocado(L.shape, centro_inferior, (largura_boca * 0.13, largura_boca * 0.035),
                            angulo, desfoque)
             + 0.5 * oval_desfocado(L.shape, centro_superior, (largura_boca * 0.08, largura_boca * 0.02),
                                    angulo, desfoque))
    L = clarear(L, gloss * mascara, GLOSS_BATOM)

    lab = np.clip(cv2.merge([L, A, B]), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def aplicar_batom(frame, pontos, tecla):
    """Escolhe o efeito certo: original (5, 6, 7) ou natural (os outros)."""
    _, cor, opacidade = BATONS[tecla]
    if tecla in BATONS_ORIGINAIS:
        return passar_batom(frame, pontos, cor, opacidade)
    return passar_batom_natural(frame, pontos, cor, opacidade)


def passar_blush(frame, pontos, cor, opacidade):
    """Blush, calculado só na região das bochechas (mais rápido)."""
    largura_rosto = np.linalg.norm(pontos[LATERAL_DIREITA] - pontos[LATERAL_ESQUERDA])
    # Todos os pontos que o blush usa (para saber onde recortar)
    indices = [LATERAL_ESQUERDA, LATERAL_DIREITA]
    for lado in BLUSH_LADOS:
        indices += lado["inicio"] + lado["luz"] + [lado["tempora"]]
    return no_recorte(frame, pontos, indices, largura_rosto * 0.45,
                      pintar_blush, cor, opacidade)


def pintar_blush(frame, pontos, cor, opacidade):
    """Blush grande e difuso na maçã do rosto, subindo até a têmpora."""
    # Tamanho do rosto: distância entre as laterais. Assim o blush
    # fica do tamanho certo, perto ou longe da câmera.
    largura_rosto = np.linalg.norm(pontos[LATERAL_DIREITA] - pontos[LATERAL_ESQUERDA])
    formato = frame.shape[:2]

    # Para cada lado: começa embaixo do olho e vai na diagonal até a têmpora.
    # Como tudo vem de pontos do rosto, o blush se move e gira com a cabeça.
    lados = []
    for lado in BLUSH_LADOS:
        inicio = pontos[lado["inicio"]].mean(axis=0)
        direcao = pontos[lado["tempora"]] - inicio
        angulo = np.degrees(np.arctan2(direcao[1], direcao[0]))
        centro = inicio + direcao * 0.30
        luz = pontos[lado["luz"]].mean(axis=0)
        lados.append((centro, angulo, luz))

    # 1) Várias camadas de ovais desfocados, somadas (pigmento acumulado):
    #    forte no centro e sumindo aos poucos, sem nenhuma borda.
    mascara = np.zeros(formato, dtype=np.float32)
    for tamanho, peso_camada in CAMADAS_BLUSH:
        eixos = (largura_rosto * 0.22 * tamanho, largura_rosto * 0.09 * tamanho)
        desfoque = largura_rosto * 0.30 * tamanho
        for centro, angulo, _ in lados:
            mascara += oval_desfocado(formato, centro, eixos, angulo, desfoque,
                                      reducao=4) * peso_camada

    # 2) Mistura a cor no LAB (mexendo pouco no brilho, a pele continua)
    peso = np.clip(mascara, 0, 1) * opacidade
    frame = aplicar_cor(frame, peso, cor, brilho=BRILHO_BLUSH)

    # 3) Iluminador: um brilho bem leve no topo da maçã do rosto
    luz = np.zeros(formato, dtype=np.float32)
    for _, angulo, ponto_luz in lados:
        luz += oval_desfocado(formato, ponto_luz, (largura_rosto * 0.09, largura_rosto * 0.035),
                              angulo, largura_rosto * 0.12, reducao=4)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = clarear(lab[:, :, 0], luz, ILUMINADOR)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def passar_sombra(frame, pontos, cor, opacidade, tem_brilho):
    """Sombra nos olhos, calculada só na região dos olhos (mais rápido)."""
    largura_rosto = np.linalg.norm(pontos[LATERAL_DIREITA] - pontos[LATERAL_ESQUERDA])
    indices = []
    for olho in OLHOS:
        indices += olho["olho"] + olho["sobrancelha"]
    return no_recorte(frame, pontos, indices, largura_rosto * 0.12,
                      pintar_sombra, cor, opacidade, tem_brilho)


def pintar_sombra(frame, pontos, cor, opacidade, tem_brilho):
    """Sombra com degradê: forte nos cílios, sumindo até a sobrancelha."""
    formato = frame.shape[:2]
    mascara = np.zeros(formato, dtype=np.float32)
    olhos_abertos = np.zeros(formato, dtype=np.uint8)

    for olho in OLHOS:
        # 1) Região da pálpebra: entre a linha dos cílios e a sobrancelha.
        #    Como vem dos pontos a cada quadro, acompanha a pálpebra ao piscar.
        poligono = pontos[olho["palpebra"] + olho["sobrancelha"]].astype(np.int32)
        regiao = np.zeros(formato, dtype=np.uint8)
        cv2.fillPoly(regiao, [poligono], 255)

        # 2) Parte aberta do olho (vai ser protegida, sem tinta)
        aberto = np.zeros(formato, dtype=np.uint8)
        cv2.fillPoly(aberto, [pontos[olho["olho"]].astype(np.int32)], 255)
        olhos_abertos = np.maximum(olhos_abertos, aberto)

        # 3) Distância de cada pixel até o olho: 0 nos cílios e cresce para cima
        distancia = cv2.distanceTransform(255 - aberto, cv2.DIST_L2, 5)
        altura = np.linalg.norm(pontos[olho["meio_sobrancelha"]] - pontos[olho["topo_olho"]])

        # 4) Duas camadas somadas (pigmento acumulado):
        #    uma curta e forte perto dos cílios + uma longa e suave até a sobrancelha
        curta = np.clip(1 - distancia / (altura * 0.7), 0, 1)
        longa = np.clip(1 - distancia / (altura * 1.3), 0, 1)
        mascara += (regiao / 255.0) * (0.5 * curta + 0.5 * longa)

    # 5) Desfoque para não ter nenhuma borda visível
    altura_media = np.mean([np.linalg.norm(pontos[o["meio_sobrancelha"]] - pontos[o["topo_olho"]])
                            for o in OLHOS])
    k = impar(altura_media * 0.5)
    mascara = cv2.GaussianBlur(mascara, (k, k), 0)

    # 6) Não pinta o olho em si: zera a máscara na parte aberta do olho
    olhos_abertos = cv2.GaussianBlur(olhos_abertos, (3, 3), 0).astype(np.float32) / 255.0
    mascara = np.clip(mascara, 0, 1) * (1 - olhos_abertos)

    # 7) Mistura a cor no LAB (textura da pele continua por baixo)
    frame = aplicar_cor(frame, mascara * opacidade, cor, brilho=BRILHO_SOMBRA)

    # 8) Sombras com brilho (champanhe): um reflexo de luz bem leve
    if tem_brilho:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = clarear(lab[:, :, 0], mascara, SHIMMER)
        frame = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return frame


# ------------------------------ PAINEL -----------------------------

# Cada aba do painel e o dicionário de cores dela (na ordem das abas)
CATEGORIAS = {"batom": BATONS, "blush": BLUSHES, "sombra": SOMBRAS}
ABAS = list(CATEGORIAS)
LARGURA_ABA = PAINEL_LARGURA // len(ABAS)


def itens_da_aba(aba):
    """Lista as teclas das cores da aba, mais None (a bolinha "sem cor")."""
    return list(CATEGORIAS[aba]) + [None]


def posicao_bolinha(i):
    """Centro (x, y) da bolinha número i, dentro do painel (3 por linha)."""
    return COLUNAS_X[i % 3], LINHA_Y0 + (i // 3) * ESPACO_Y


def desenhar_icone_batom(img, cx, cy, cor):
    """Desenha um batom: tubo escuro, anel dourado e ponta inclinada."""
    cv2.rectangle(img, (cx - 7, cy + 2), (cx + 7, cy + 18), (60, 60, 60), -1)   # tubo
    cv2.rectangle(img, (cx - 7, cy - 3), (cx + 7, cy + 2), (60, 170, 210), -1)  # anel
    ponta = np.array([[cx - 5, cy - 3], [cx + 5, cy - 3],
                      [cx + 5, cy - 13], [cx - 5, cy - 19]], dtype=np.int32)
    cv2.fillPoly(img, [ponta], cor)                                             # bala


def desenhar_icone_pincel(img, cx, cy, cor):
    """Desenha um pincel de blush: cabo, virola metálica e cerdas coloridas."""
    cv2.rectangle(img, (cx - 3, cy + 3), (cx + 3, cy + 20), (40, 70, 110), -1)  # cabo
    cv2.rectangle(img, (cx - 5, cy - 3), (cx + 5, cy + 3), (190, 190, 190), -1) # virola
    cv2.ellipse(img, (cx, cy - 11), (9, 10), 0, 0, 360, cor, -1)               # cerdas


def desenhar_icone_sombra(img, cx, cy, cor):
    """Desenha um olho com a pálpebra pintada da cor da sombra."""
    cv2.ellipse(img, (cx, cy + 2), (14, 13), 0, 180, 360, cor, -1)              # sombra
    cv2.ellipse(img, (cx, cy + 4), (13, 6), 0, 0, 360, (235, 235, 235), -1)     # olho
    cv2.circle(img, (cx, cy + 4), 4, (70, 45, 30), -1)                           # íris
    cv2.ellipse(img, (cx, cy + 4), (13, 6), 0, 0, 360, (30, 30, 30), 1, cv2.LINE_AA)  # contorno


# Ícone e nome mostrados em cada aba
ICONES = {"batom": desenhar_icone_batom, "blush": desenhar_icone_pincel,
          "sombra": desenhar_icone_sombra}
TITULOS = {"batom": "Batom", "blush": "Blush", "sombra": "Sombra"}


def escrever(img, texto, posicao, tamanho=0.45, cor=BRANCO):
    """Escreve um texto pequeno no painel."""
    cv2.putText(img, texto, posicao, cv2.FONT_HERSHEY_SIMPLEX, tamanho,
                cor, 1, cv2.LINE_AA)


def desenhar_painel(altura, estado):
    """Monta a imagem do painel lateral com abas e bolinhas de cor."""
    painel = np.full((altura, PAINEL_LARGURA, 3), COR_FUNDO, dtype=np.uint8)

    # --- Abas (Batom | Blush | Sombra), cada uma com seu ícone ---
    for i, aba in enumerate(ABAS):
        x0 = i * LARGURA_ABA
        fundo = COR_ABA_ATIVA if estado["aba"] == aba else COR_ABA_INATIVA
        cv2.rectangle(painel, (x0, 0), (x0 + LARGURA_ABA - 1, ABA_ALTURA), fundo, -1)
        # O ícone fica da cor escolhida (ou cinza, se estiver "sem cor")
        tecla = estado[aba]
        cor_icone = CATEGORIAS[aba][tecla][1] if tecla else CINZA
        ICONES[aba](painel, x0 + 18, 30, cor_icone)
        escrever(painel, TITULOS[aba], (x0 + 36, 36), 0.42)
        # Linha clara embaixo da aba ativa
        if estado["aba"] == aba:
            cv2.line(painel, (x0, ABA_ALTURA), (x0 + LARGURA_ABA - 1, ABA_ALTURA), BRANCO, 2)

    # --- Bolinhas de cor da aba ativa ---
    cores = CATEGORIAS[estado["aba"]]
    escolhida = estado[estado["aba"]]  # tecla da cor escolhida nessa aba
    for i, tecla in enumerate(itens_da_aba(estado["aba"])):
        centro = posicao_bolinha(i)
        if tecla is None:
            # Bolinha "sem cor": cinza com um X
            cv2.circle(painel, centro, RAIO_BOLINHA, (90, 90, 90), -1)
            d = RAIO_BOLINHA // 2
            cx, cy = centro
            cv2.line(painel, (cx - d, cy - d), (cx + d, cy + d), BRANCO, 2, cv2.LINE_AA)
            cv2.line(painel, (cx - d, cy + d), (cx + d, cy - d), BRANCO, 2, cv2.LINE_AA)
        else:
            cv2.circle(painel, centro, RAIO_BOLINHA, cores[tecla][1], -1, cv2.LINE_AA)
            # Tecla de atalho, pequena, embaixo da bolinha
            escrever(painel, tecla, (centro[0] - 4, centro[1] + RAIO_BOLINHA + 12), 0.35, CINZA)
        # Contorno branco na bolinha escolhida
        if tecla == escolhida:
            cv2.circle(painel, centro, RAIO_BOLINHA + 4, BRANCO, 2, cv2.LINE_AA)

    # --- Nome das cores escolhidas ---
    y = LINHA_Y0 + 3 * ESPACO_Y
    for i, aba in enumerate(ABAS):
        tecla = estado[aba]
        nome = CATEGORIAS[aba][tecla][0] if tecla else "sem cor"
        escrever(painel, f"{TITULOS[aba]}: {nome}", (10, y + i * 22))

    # --- Atalhos de teclado ---
    escrever(painel, "Atalhos: 1-8 batom, 0 tira", (10, altura - 64), 0.38, CINZA)
    escrever(painel, "a s d f g h blush, b liga/desl.", (10, altura - 46), 0.38, CINZA)
    escrever(painel, "z x c v n m j sombra, o liga/desl.", (10, altura - 28), 0.38, CINZA)
    escrever(painel, "q: sair", (10, altura - 10), 0.38, CINZA)
    return painel


def ao_clicar(evento, x, y, flags, estado):
    """Chamada pelo OpenCV a cada evento do mouse na janela."""
    if evento != cv2.EVENT_LBUTTONDOWN:
        return

    # O painel fica à direita do vídeo: converte x para dentro do painel
    x = x - estado["largura_video"]
    if x < 0:
        return  # clicou no vídeo, não no painel

    # Clique nas abas
    if y <= ABA_ALTURA:
        indice = min(x // LARGURA_ABA, len(ABAS) - 1)  # qual aba foi clicada
        estado["aba"] = ABAS[indice]
        return

    # Clique numa bolinha: confere se o clique está dentro do círculo
    for i, tecla in enumerate(itens_da_aba(estado["aba"])):
        cx, cy = posicao_bolinha(i)
        if (x - cx) ** 2 + (y - cy) ** 2 <= RAIO_BOLINHA ** 2:
            escolher(estado, estado["aba"], tecla)
            return


def escolher(estado, aba, tecla):
    """Aplica a cor escolhida (tecla) ou tira a maquiagem (None)."""
    estado[aba] = tecla
    if aba == "blush" and tecla is not None:
        estado["ultimo_blush"] = tecla   # lembrado pela tecla "b"
    if aba == "sombra" and tecla is not None:
        estado["ultima_sombra"] = tecla  # lembrado pela tecla "o"


def tratar_tecla(estado, tecla):
    """Atalhos de teclado. Devolve False quando é para sair."""
    if tecla == "q":
        return False
    elif tecla in BATONS:                  # 1 a 8: troca o batom
        escolher(estado, "batom", tecla)
        estado["aba"] = "batom"
    elif tecla == "0":                     # 0: tira o batom
        escolher(estado, "batom", None)
        estado["aba"] = "batom"
    elif tecla in BLUSHES:                 # a, s, d, f, g, h: troca o blush
        escolher(estado, "blush", tecla)
        estado["aba"] = "blush"
    elif tecla == "b":                     # b: liga/desliga o blush
        novo = None if estado["blush"] else estado["ultimo_blush"]
        escolher(estado, "blush", novo)
        estado["aba"] = "blush"
    elif tecla in SOMBRAS:                 # z, x, c, v, n, m, j: troca a sombra
        escolher(estado, "sombra", tecla)
        estado["aba"] = "sombra"
    elif tecla == "o":                     # o: liga/desliga a sombra
        novo = None if estado["sombra"] else estado["ultima_sombra"]
        escolher(estado, "sombra", novo)
        estado["aba"] = "sombra"
    return True


# ------------------------------- MAIN ------------------------------

def main():
    detector = criar_detector()

    # Abre a webcam (0 = câmera padrão do computador)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Não foi possível abrir a webcam.")
        return

    # Tudo que o usuário pode mudar fica neste dicionário, que é
    # compartilhado com a função do mouse
    estado = {
        "aba": "batom",         # aba aberta no painel
        "batom": "1",           # tecla do batom escolhido (None = sem batom)
        "blush": "a",           # tecla do blush escolhido (None = sem blush)
        "ultimo_blush": "a",    # para a tecla "b" religar o mesmo blush
        "sombra": "z",          # tecla da sombra escolhida (None = sem sombra)
        "ultima_sombra": "z",   # para a tecla "o" religar a mesma sombra
        "largura_video": 0,     # usado para saber onde o painel começa
    }

    # Cria a janela e liga a função que trata os cliques do mouse
    cv2.namedWindow(JANELA)
    cv2.setMouseCallback(JANELA, ao_clicar, estado)

    while True:
        # Lê um quadro (frame) da câmera
        ok, frame = camera.read()
        if not ok:
            print("Falha ao ler a imagem da webcam.")
            break

        # Espelha a imagem, para parecer um espelho de verdade
        frame = cv2.flip(frame, 1)
        estado["largura_video"] = frame.shape[1]

        # Procura o rosto e aplica a maquiagem escolhida
        pontos = detectar_rosto(detector, frame)
        if pontos is not None:
            if estado["sombra"]:
                _, cor, opacidade, tem_brilho = SOMBRAS[estado["sombra"]]
                frame = passar_sombra(frame, pontos, cor, opacidade, tem_brilho)
            if estado["blush"]:
                _, cor, opacidade = BLUSHES[estado["blush"]]
                frame = passar_blush(frame, pontos, cor, opacidade)
            if estado["batom"]:
                frame = aplicar_batom(frame, pontos, estado["batom"])

        # Junta o vídeo com o painel ao lado e mostra
        painel = desenhar_painel(frame.shape[0], estado)
        cv2.imshow(JANELA, np.hstack([frame, painel]))

        # Espera 1 ms por uma tecla (255 = nenhuma tecla)
        codigo = cv2.waitKey(1) & 0xFF
        if codigo != 255 and not tratar_tecla(estado, chr(codigo).lower()):
            break

    # Libera a câmera e fecha as janelas
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

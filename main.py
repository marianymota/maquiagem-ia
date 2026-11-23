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
#
# Batons: tecla -> (nome, cor BGR, opacidade)
# A opacidade vai de 0 (invisível) a 1 (cor total). Entre 0.20 e 0.35
# a textura natural da boca continua aparecendo.
BATONS = {
    "1": ("Nude rosado", (118, 122, 176), 0.30),
    "2": ("Rosa-cha", (125, 120, 186), 0.30),
    "3": ("Coral", (95, 110, 212), 0.30),
    "4": ("Vermelho classico", (35, 20, 170), 0.35),
    "5": ("Cereja", (55, 15, 160), 0.35),
    "6": ("Vinho/berry", (45, 20, 110), 0.35),
    "7": ("Marrom-chocolate", (45, 55, 105), 0.35),
    "8": ("Malva", (130, 100, 160), 0.30),
}

# Blushes: tecla -> (nome, cor BGR, opacidade)
BLUSHES = {
    "a": ("Rosa-pessego", (140, 150, 240), 0.30),
    "s": ("Rose", (145, 130, 220), 0.30),
    "d": ("Coral suave", (110, 128, 240), 0.30),
    "f": ("Terracota", (75, 100, 190), 0.28),
    "g": ("Malva", (150, 120, 180), 0.30),
    "h": ("Berry suave", (110, 80, 170), 0.28),
}

# Quanto o batom pode escurecer/clarear a boca (0 = só muda a cor,
# 1 = brilho igual ao da cor do batom). Um pouco ajuda tons escuros como
# vinho e chocolate a aparecerem, sem apagar a textura dos lábios.
BRILHO_BATOM = 0.7
# ====================================================================

# O Face Mesh encontra 478 pontos no rosto. Estes são os números (índices)
# dos pontos que formam o contorno dos lábios, já em ordem, dando a volta.
LABIOS_EXTERNO = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                  291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
# Contorno de dentro da boca (para não pintar os dentes com a boca aberta)
LABIOS_INTERNO = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                  308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# Pontos usados para o blush
MACA_ESQUERDA = 205   # meio da maçã do rosto (um lado)
MACA_DIREITA = 425    # meio da maçã do rosto (outro lado)
LATERAL_ESQUERDA = 234  # borda do rosto, para medir a largura
LATERAL_DIREITA = 454
OLHO_ESQUERDO = 33    # cantos dos olhos, para medir a inclinação da cabeça
OLHO_DIREITO = 263


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


def passar_blush(frame, pontos, cor, opacidade):
    """Aplica blush nas maçãs do rosto como duas manchas ovais bem suaves."""
    # Tamanho do rosto: distância entre as laterais. Assim o blush
    # fica do tamanho certo, perto ou longe da câmera.
    largura_rosto = np.linalg.norm(pontos[LATERAL_DIREITA] - pontos[LATERAL_ESQUERDA])

    # Inclinação da cabeça (em graus), pela linha entre os olhos.
    # Assim o oval gira junto quando a cabeça inclina.
    dx, dy = pontos[OLHO_DIREITO] - pontos[OLHO_ESQUERDO]
    angulo = np.degrees(np.arctan2(dy, dx))

    # Tamanho do oval: mais largo do que alto
    eixos = (int(largura_rosto * 0.13), int(largura_rosto * 0.08))

    # 1) Desenha os dois ovais brancos numa máscara preta
    mascara = np.zeros(frame.shape[:2], dtype=np.uint8)
    for indice in (MACA_ESQUERDA, MACA_DIREITA):
        centro = tuple(int(v) for v in pontos[indice])
        cv2.ellipse(mascara, centro, eixos, angulo, 0, 360, 255, -1)

    # 2) Desfoque bem grande, para o blush "sumir" na pele sem borda
    k = int(largura_rosto * 0.25) // 2 * 2 + 1
    mascara = cv2.GaussianBlur(mascara, (k, k), 0)

    # 3) Pesos de 0 a 1 e mistura só da cor (sem mexer no brilho da pele)
    peso = mascara.astype(np.float32) / 255.0 * opacidade
    return aplicar_cor(frame, peso, cor, brilho=0.0)


def escrever(frame, texto, posicao, tamanho=0.5):
    """Escreve um texto branco com contorno preto (legível em qualquer fundo)."""
    cv2.putText(frame, texto, posicao, cv2.FONT_HERSHEY_SIMPLEX, tamanho,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, texto, posicao, cv2.FONT_HERSHEY_SIMPLEX, tamanho,
                (255, 255, 255), 1, cv2.LINE_AA)


def main():
    detector = criar_detector()

    # Abre a webcam (0 = câmera padrão do computador)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Não foi possível abrir a webcam.")
        return

    # Escolhas atuais (podem ser trocadas pelo teclado)
    batom_atual = "1"     # tecla do batom; None = sem batom
    blush_atual = "a"     # tecla do blush
    blush_ligado = True   # a tecla "b" liga/desliga o blush

    while True:
        # Lê um quadro (frame) da câmera
        ok, frame = camera.read()
        if not ok:
            print("Falha ao ler a imagem da webcam.")
            break

        # Espelha a imagem, para parecer um espelho de verdade
        frame = cv2.flip(frame, 1)

        # Procura o rosto e aplica a maquiagem escolhida
        pontos = detectar_rosto(detector, frame)
        if pontos is not None:
            if blush_ligado:
                _, cor, opacidade = BLUSHES[blush_atual]
                frame = passar_blush(frame, pontos, cor, opacidade)
            if batom_atual is not None:
                _, cor, opacidade = BATONS[batom_atual]
                frame = passar_batom(frame, pontos, cor, opacidade)

        # Mostra a combinação atual e as instruções
        nome_batom = BATONS[batom_atual][0] if batom_atual else "sem batom"
        nome_blush = BLUSHES[blush_atual][0] if blush_ligado else "sem blush"
        escrever(frame, f"Batom: {nome_batom} | Blush: {nome_blush}", (10, 25))
        escrever(frame, "1-8 batom | 0 tira batom | a s d f g h blush | "
                        "b liga/desliga blush | q sai", (10, 45), 0.4)

        # Mostra o quadro numa janela
        cv2.imshow("Try-on de Maquiagem", frame)

        # Espera 1 ms por uma tecla (255 = nenhuma tecla)
        codigo = cv2.waitKey(1) & 0xFF
        if codigo == 255:
            continue
        tecla = chr(codigo).lower()

        if tecla == "q":
            break                      # sai do programa
        elif tecla in BATONS:
            batom_atual = tecla        # 1 a 8: troca o batom
        elif tecla == "0":
            batom_atual = None         # 0: tira o batom
        elif tecla in BLUSHES:
            blush_atual = tecla        # a, s, d, f, g, h: troca o blush
            blush_ligado = True
        elif tecla == "b":
            blush_ligado = not blush_ligado  # b: liga/desliga o blush

    # Libera a câmera e fecha as janelas
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

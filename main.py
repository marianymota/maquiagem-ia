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
    "1": ("Nude rosado", (130, 140, 200), 0.40),
    "2": ("Rosa-cha", (125, 110, 190), 0.45),
    "3": ("Coral", (70, 95, 240), 0.45),
    "4": ("Vermelho classico", (30, 20, 200), 0.50),
    "5": ("Cereja escuro", (40, 10, 130), 0.50),
    "6": ("Vinho/berry", (70, 25, 105), 0.50),
    "7": ("Marrom-chocolate", (35, 50, 95), 0.50),
    "8": ("Malva", (140, 95, 165), 0.45),
}

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
BRILHO_BATOM = 0.8
BRILHO_BLUSH = 0.15
# ====================================================================

# O Face Mesh encontra 478 pontos no rosto. Estes são os números (índices)
# dos pontos que formam o contorno dos lábios, já em ordem, dando a volta.
LABIOS_EXTERNO = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                  291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
# Contorno de dentro da boca (para não pintar os dentes com a boca aberta)
LABIOS_INTERNO = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                  308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# Pontos usados para o blush, um grupo para cada lado do rosto:
# - "maca": pontos cuja média é o centro do blush (no alto da maçã do rosto)
# - "tempora": ponto para onde o blush se alonga, como na maquiagem real
BLUSH_LADOS = [
    {"maca": [50, 116], "tempora": 162},
    {"maca": [280, 345], "tempora": 389},
]
LATERAL_ESQUERDA = 234  # bordas do rosto, para medir a largura
LATERAL_DIREITA = 454

# ========================= PAINEL CLICÁVEL ==========================
JANELA = "Try-on de Maquiagem"
PAINEL_LARGURA = 240        # largura do painel ao lado do vídeo
ABA_ALTURA = 60             # altura dos botões das abas
RAIO_BOLINHA = 22           # tamanho das bolinhas de cor
COLUNAS_X = [45, 120, 195]  # posição x das 3 colunas de bolinhas
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


def passar_blush(frame, pontos, cor, opacidade):
    """Aplica blush no alto das maçãs do rosto, alongado até as têmporas."""
    # Tamanho do rosto: distância entre as laterais. Assim o blush
    # fica do tamanho certo, perto ou longe da câmera.
    largura_rosto = np.linalg.norm(pontos[LATERAL_DIREITA] - pontos[LATERAL_ESQUERDA])

    # Tamanho do oval: comprido (na direção da têmpora) e estreito
    eixos = (int(largura_rosto * 0.15), int(largura_rosto * 0.065))

    # 1) Desenha um oval branco em cada lado do rosto, numa máscara preta
    mascara = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lado in BLUSH_LADOS:
        # Centro: média dos pontos do alto da maçã do rosto
        centro = pontos[lado["maca"]].mean(axis=0)
        # Direção: da maçã do rosto até a têmpora (a diagonal do blush).
        # Como usa pontos do rosto, o oval gira junto com a cabeça.
        dx, dy = pontos[lado["tempora"]] - centro
        angulo = np.degrees(np.arctan2(dy, dx))
        # Puxa o centro um pouco em direção à têmpora
        centro = centro + np.array([dx, dy]) * 0.08
        cv2.ellipse(mascara, (int(centro[0]), int(centro[1])), eixos,
                    angulo, 0, 360, 255, -1)

    # 2) Desfoque bem grande, para o blush "sumir" na pele sem borda
    k = int(largura_rosto * 0.25) // 2 * 2 + 1
    mascara = cv2.GaussianBlur(mascara, (k, k), 0)

    # 3) Pesos de 0 a 1 e mistura da cor, mexendo pouco no brilho da pele
    peso = mascara.astype(np.float32) / 255.0 * opacidade
    return aplicar_cor(frame, peso, cor, brilho=BRILHO_BLUSH)


# ------------------------------ PAINEL -----------------------------

def itens_da_aba(aba):
    """Lista as teclas das cores da aba, mais None (a bolinha "sem cor")."""
    cores = BATONS if aba == "batom" else BLUSHES
    return list(cores) + [None]


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


def escrever(img, texto, posicao, tamanho=0.45, cor=BRANCO):
    """Escreve um texto pequeno no painel."""
    cv2.putText(img, texto, posicao, cv2.FONT_HERSHEY_SIMPLEX, tamanho,
                cor, 1, cv2.LINE_AA)


def desenhar_painel(altura, estado):
    """Monta a imagem do painel lateral com abas e bolinhas de cor."""
    painel = np.full((altura, PAINEL_LARGURA, 3), COR_FUNDO, dtype=np.uint8)
    meio = PAINEL_LARGURA // 2

    # --- Abas (Batom | Blush) ---
    cor_batom = BATONS[estado["batom"]][1] if estado["batom"] else CINZA
    cor_blush = BLUSHES[estado["blush"]][1] if estado["blush"] else CINZA
    for aba, x0 in (("batom", 0), ("blush", meio)):
        fundo = COR_ABA_ATIVA if estado["aba"] == aba else COR_ABA_INATIVA
        cv2.rectangle(painel, (x0, 0), (x0 + meio - 1, ABA_ALTURA), fundo, -1)
        if aba == "batom":
            desenhar_icone_batom(painel, x0 + 22, 30, cor_batom)
            escrever(painel, "Batom", (x0 + 40, 36))
        else:
            desenhar_icone_pincel(painel, x0 + 22, 30, cor_blush)
            escrever(painel, "Blush", (x0 + 40, 36))
    # Linha clara embaixo da aba ativa
    x0 = 0 if estado["aba"] == "batom" else meio
    cv2.line(painel, (x0, ABA_ALTURA), (x0 + meio - 1, ABA_ALTURA), BRANCO, 2)

    # --- Bolinhas de cor da aba ativa ---
    cores = BATONS if estado["aba"] == "batom" else BLUSHES
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
    nome_batom = BATONS[estado["batom"]][0] if estado["batom"] else "sem batom"
    nome_blush = BLUSHES[estado["blush"]][0] if estado["blush"] else "sem blush"
    y = LINHA_Y0 + 3 * ESPACO_Y
    escrever(painel, f"Batom: {nome_batom}", (10, y))
    escrever(painel, f"Blush: {nome_blush}", (10, y + 22))

    # --- Atalhos de teclado ---
    escrever(painel, "Atalhos: 1-8 batom, 0 tira", (10, altura - 46), 0.38, CINZA)
    escrever(painel, "a s d f g h blush, b liga/desl.", (10, altura - 28), 0.38, CINZA)
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
        estado["aba"] = "batom" if x < PAINEL_LARGURA // 2 else "blush"
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
        estado["ultimo_blush"] = tecla  # lembrado pela tecla "b"


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
            if estado["blush"]:
                _, cor, opacidade = BLUSHES[estado["blush"]]
                frame = passar_blush(frame, pontos, cor, opacidade)
            if estado["batom"]:
                _, cor, opacidade = BATONS[estado["batom"]]
                frame = passar_batom(frame, pontos, cor, opacidade)

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

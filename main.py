# Try-on virtual de maquiagem com a webcam
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

# O Face Mesh encontra 478 pontos no rosto. Estes são os números (índices)
# dos pontos que formam o contorno dos lábios, já em ordem, dando a volta.
LABIOS_EXTERNO = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                  291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
# Contorno de dentro da boca (para não pintar os dentes com a boca aberta)
LABIOS_INTERNO = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                  308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# Cores de batom disponíveis. Cada tecla tem um nome e uma cor.
# Atenção: o OpenCV usa a ordem BGR (azul, verde, vermelho), não RGB!
CORES = {
    "1": ("Vermelho", (40, 20, 200)),
    "2": ("Rosa", (147, 80, 230)),
    "3": ("Vinho", (60, 20, 110)),
}


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


def pontos_dos_labios(detector, frame):
    """Detecta o rosto e devolve os contornos dos lábios em pixels.

    Retorna (externo, interno) como arrays NumPy, ou (None, None)
    se nenhum rosto for encontrado.
    """
    # O OpenCV usa cores em BGR, mas o MediaPipe espera RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    resultado = detector.detect(imagem)

    if not resultado.face_landmarks:
        return None, None

    rosto = resultado.face_landmarks[0]
    altura, largura = frame.shape[:2]

    # Os pontos vêm "normalizados" (de 0 a 1); multiplicamos pelo
    # tamanho da imagem para converter em pixels
    def converter(indices):
        return np.array(
            [(int(rosto[i].x * largura), int(rosto[i].y * altura)) for i in indices],
            dtype=np.int32,
        )

    return converter(LABIOS_EXTERNO), converter(LABIOS_INTERNO)


def pintar_labios(frame, externo, interno, cor, opacidade=0.4):
    """Pinta a área dos lábios com uma cor semitransparente (o "batom").

    cor: cor no formato BGR do OpenCV, ex.: (0, 0, 255) = vermelho
    opacidade: 0 = invisível, 1 = cor sólida
    """
    # 1) Cria uma "máscara": imagem preta onde a área dos lábios fica branca
    mascara = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mascara, [externo], 255)  # preenche o contorno externo
    cv2.fillPoly(mascara, [interno], 0)    # "apaga" o buraco da boca

    # 2) Desfoca a borda da máscara, para o batom não ficar "recortado"
    mascara = cv2.GaussianBlur(mascara, (7, 7), 0)

    # 3) Transforma a máscara em pesos de 0 a 1 e aplica a opacidade
    peso = (mascara.astype(np.float32) / 255.0) * opacidade
    peso = peso[:, :, np.newaxis]  # uma dimensão a mais, para os 3 canais de cor

    # 4) Mistura: onde o peso é 0 fica a imagem original; onde é maior,
    #    a cor do batom aparece por cima
    camada_cor = np.full_like(frame, cor, dtype=np.uint8)
    misturado = frame * (1 - peso) + camada_cor * peso
    return misturado.astype(np.uint8)


def main():
    detector = criar_detector()

    # Abre a webcam (0 = câmera padrão do computador)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Não foi possível abrir a webcam.")
        return

    # Começa com a primeira cor da lista
    nome_cor, cor = CORES["1"]

    while True:
        # Lê um quadro (frame) da câmera
        ok, frame = camera.read()
        if not ok:
            print("Falha ao ler a imagem da webcam.")
            break

        # Espelha a imagem, para parecer um espelho de verdade
        frame = cv2.flip(frame, 1)

        # Procura os lábios e, se achar um rosto, passa o batom
        externo, interno = pontos_dos_labios(detector, frame)
        if externo is not None:
            frame = pintar_labios(frame, externo, interno, cor)

        # Escreve na tela a cor atual e as instruções
        cv2.putText(frame, f"Batom: {nome_cor}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
        cv2.putText(frame, "Teclas 1, 2, 3: trocar cor | q: sair", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mostra o quadro numa janela
        cv2.imshow("Try-on de Maquiagem", frame)

        # Espera 1 ms por uma tecla
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break  # "q" sai do programa
        elif chr(tecla) in CORES:
            nome_cor, cor = CORES[chr(tecla)]  # 1, 2 ou 3 troca a cor

    # Libera a câmera e fecha as janelas
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

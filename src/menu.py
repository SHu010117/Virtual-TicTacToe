import math

import cv2
import mediapipe as mp
import pygame
import numpy as np

SUPERFTIMEPATH = '../assets/fonts/super-funtime-font/SuperFuntime-3zpLX.ttf'
THEGLOBEFONTPATH = '../assets/fonts/the-globe-font/TheGlobePersonalUseBold-2ORlw.ttf'
NINTENDOPATH = '../assets/fonts/ro-spritendo-font/RoSpritendoSemiboldBeta-vmVwZ.otf'
PIXELPATH = '../assets/fonts/public-pixel-font/PublicPixel-E447g.ttf'

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DGREEN = (24, 60, 37)
DGREEN2 = (24, 60, 37)

# Costante di errore
EPSILON = 0

# Inizializzazione di Pygame
pygame.init()
pygame.mixer.init()

# Carica e riproduci la musica di sottofondo
# pygame.mixer.music.load('../assets/music/arcade-party-173553.mp3')
# pygame.mixer.music.play(-1)  # loop

# Video background
# video_capture = cv2.VideoCapture('../assets/videos/pika.mp4')
# frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

# Creazione Window
WIDTH, HEIGHT = 1000, 680
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe')
background_image = pygame.image.load('../assets/images/background/prova3.jpeg')
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))
grid_img = pygame.image.load('../assets/images/background/tic.jpg')
grid_img = pygame.transform.scale(grid_img, (WIDTH, HEIGHT))

FPS = 60
CLOCK = pygame.time.Clock()

# Inizializzazione cv2
cap = cv2.VideoCapture(0)

# Inizializzazione di Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

draws = [[]]
drawNumber = -1
drawStart = False
def draw_game(indexpos, draw):
    if draw:
        pygame.draw.circle(WIN, (0, 255, 0), indexpos, 7)
    elif indexpos:
        pygame.draw.circle(WIN, (255, 0, 0), indexpos, 7)

    for i in range(len(draws)):
        for j in range(len(draws[i])):
            if j != 0:
                pygame.draw.line(WIN, (255, 0, 255), draws[i][j-1], draws[i][j], 7)



    pygame.display.flip()


def draw_menu(show_text, indexpos):
    # WIN.fill((245,255,255))

    # Scritta principale
    font = pygame.font.Font(PIXELPATH, 78)
    text = font.render('TIC-TAC-TOE', True, DGREEN)
    WIN.blit(text, (WIDTH // 2 - text.get_width() // 2, (HEIGHT // 8)))

    # Scritta secondaria
    if show_text:
        font = pygame.font.Font(PIXELPATH, 24)
        text = font.render('Pollice in sù per iniziare', True, DGREEN2)
        WIN.blit(text, (WIDTH // 2 - text.get_width() // 2, (HEIGHT // 4) + 65))

    if indexpos:
        pygame.draw.circle(WIN, (255, 0, 0), indexpos, 6)

    # Aggiorna l'intera schermata
    pygame.display.flip()


def fingers_up(hand_landmarks):
    fingers = []
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[
        mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(True)
    else:
        fingers.append(False)

    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_PIP].y)
    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.RING_FINGER_PIP].y)
    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.PINKY_PIP].y)

    return fingers


def is_thumb_up_and_fist_closed(hand_landmarks):
    # Punti chiave dita
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    # Verificare se il pollice è alzato
    is_thumb_up = thumb_tip.y < thumb_mcp.y

    # Verificare se il pugno è chiusto:
    is_fist_closed = (
            (min(thumb_mcp.x, index_mcp.x) - EPSILON <= index_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON) and
            (min(thumb_mcp.x, index_mcp.x) - EPSILON <= middle_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON) and
            (min(thumb_mcp.x, index_mcp.x) - EPSILON <= ring_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON) and
            (min(thumb_mcp.x, index_mcp.x) - EPSILON <= pinky_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON)
    )

    return is_thumb_up and is_fist_closed




menu = True
running = True
show_text = True
last_toggle_time = pygame.time.get_ticks()


# Main loop
while running:
    current_time = pygame.time.get_ticks()
    if current_time - last_toggle_time > 250:  # 0.25 s
        show_text = not show_text
        last_toggle_time = current_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()

    # Logica del gioco
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    res = hands.process(frame)
    indexpos = None
    draw = False
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            # Controllo se il pollice è alzato e il pugno chiuso
            if is_thumb_up_and_fist_closed(hand_landmarks):
                print("Pollice alzato")
                menu = False

            # Controllo se l'indice è alzato e prendo la posizione.
            fingers = fingers_up(hand_landmarks)

            if fingers[1]:
                ratio_x_to_pixel = lambda x: math.ceil(x * WIDTH)
                ratio_y_to_pixel = lambda y: math.ceil(y * HEIGHT)
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # In questo modo posso spostare il pallino verso il basso senza che scompaia o faccia cose strane. (Da sistemare)
                cx = np.interp(int(ratio_x_to_pixel(index_tip.x)), [150, WIDTH - 150], [0, WIDTH])
                cy = np.interp(int(ratio_y_to_pixel(index_tip.y)), [150, HEIGHT - 150], [0, HEIGHT])

                indexpos = (cx, cy)



            if fingers[1] and not fingers[2]:
                if not drawStart:
                    drawStart = True
                    drawNumber += 1
                    draws.append([])
                draws[drawNumber].append(indexpos)
                draw = True
            else:
                drawStart = False



    if menu:
        '''
        ret, video_frame = video_capture.read()
        if not ret:
            # Se il video è terminato, ricomincia dall'inizio
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, video_frame = video_capture.read()
        video_frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        video_frame_pygame = pygame.image.frombuffer(video_frame_rgb.tobytes(), video_frame_rgb.shape[1::-1], 'RGB')
        video_frame_pygame = pygame.transform.scale(video_frame_pygame, (WIDTH, HEIGHT))
        WIN.blit(video_frame_pygame, (0, 0))
        '''
        WIN.blit(background_image, (0, 0))
        draw_menu(show_text, indexpos)

    else:
        WIN.blit(grid_img, (0, 0))
        draw_game(indexpos, draw)

        #pygame.display.flip()

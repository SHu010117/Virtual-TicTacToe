import math
import cv2
import mediapipe as mp
import pygame
import numpy as np
from menu import draw_menu
from game import draw_game
from game import check_winner

SUPERFTIMEPATH = '../assets/fonts/super-funtime-font/SuperFuntime-3zpLX.ttf'
THEGLOBEFONTPATH = '../assets/fonts/the-globe-font/TheGlobePersonalUseBold-2ORlw.ttf'
NINTENDOPATH = '../assets/fonts/ro-spritendo-font/RoSpritendoSemiboldBeta-vmVwZ.otf'

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GREEN = (24, 60, 37)
DARK_GREEN2 = (24, 60, 37)

# Error constant
EPSILON = 0

# Pygame initialization
pygame.init()
pygame.mixer.init()

# Carica e riproduci la musica di sottofondo
# pygame.mixer.music.load('../assets/music/arcade-party-173553.mp3')
# pygame.mixer.music.play(-1)  # loop


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

# Inizializzazione dei 'disegni'
draws = [[]]
drawNumber = 0
drawStart = False
startCell = None

grid_array = [["", "", ""], ["", "", ""], ["", "", ""]]
chars = ["O", "X"]


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
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

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

x_coordinates = [192, 391, 610, 784]
y_coordinates = [53, 220, 420, 640]
def get_boundaries(index, xs, ys):
    i = index // 3
    j = index % 3
    return [xs[j], xs[j+1], ys[i], ys[i+1]]

def get_cell(xy):
    global turn
    if xy[0] <= 391 and xy[1] <= 220:
        return 0
    elif xy[0] <= 610 and xy[1] <= 220:
        return 1
    elif xy[0] > 610 and xy[1] <= 220:
        return 2
    elif xy[0] <= 391 and xy[1] <= 420:
        return 3
    elif xy[0] <= 610 and xy[1] <= 420:
        return 4
    elif xy[0] > 610 and xy[1] <= 420:
        return 5
    elif xy[0] <= 391 and xy[1] > 420:
        return 6
    elif xy[0] <= 610 and xy[1] > 420:
        return 7
    else:
        return 8
    
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
    index_pos = None
    draw = False
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            # Controllo se il pollice è alzato e il pugno chiuso
            if is_thumb_up_and_fist_closed(hand_landmarks):
                menu = False

            # Controllo se l'indice è alzato e prendo la posizione.
            fingers = fingers_up(hand_landmarks)

            if fingers[1]:
                ratio_x_to_pixel = lambda x: math.ceil(x * WIDTH)
                ratio_y_to_pixel = lambda y: math.ceil(y * HEIGHT)
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # In questo modo posso spostare il pallino verso il basso senza che scompaia o faccia cose strane. (Da sistemare)
                cx = float(np.interp(int(ratio_x_to_pixel(index_tip.x)), [150, WIDTH - 150], [0, WIDTH]))
                cy = float(np.interp(int(ratio_y_to_pixel(index_tip.y)), [150, HEIGHT - 150], [0, HEIGHT]))
                index_pos = (cx, cy)

            if fingers[1] and not fingers[2] and not menu:
                if not drawStart:
                    startCell = get_cell(index_pos)
                    drawStart = True
                    drawNumber += 1
                    draws.append([])
                boundaries = get_boundaries(startCell, x_coordinates, y_coordinates)
                indexpos = min(max(indexpos[0], boundaries[0]), boundaries[1]), min(max(indexpos[1], boundaries[2]), boundaries[3])
                draws[drawNumber].append(index_pos)
                draw = True
            else:
                drawStart = False

            if fingers == [False, True, True, True, False]:
                # print("ok")
                if draws: 
                    if drawNumber >= 0:
                        print(drawNumber)
                        if not Erasing:
                            draws.pop()
                            drawNumber -= 1
                            Erasing = True
            else:
                Erasing = False

    if menu:
        WIN.blit(background_image, (0, 0))
        # Disegno punto indice
        if index_pos:
            pygame.draw.circle(WIN, (255, 0, 0), index_pos, 6)
        draw_menu(WIN, show_text, WIDTH, HEIGHT)

    else:
        WIN.blit(grid_img, (0, 0))
        draw_game(WIN, index_pos, draw, grid_array, chars, startCell, draws)
        check_winner(grid_array)

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
drawNumber = -1
drawStart = False


intercection_points = [(373,182),(583,182),(373,347),(583,347)]
chars = ["O", "X"]
turn = 0
grid_array = [["","",""],["","",""],["","",""]]
ratio_x_to_pixel = lambda x: math.ceil(x * WIDTH)
ratio_y_to_pixel = lambda y: math.ceil(y * HEIGHT)

def check_cell(arr, xy, chars):
    global turn
    if xy[0] <= 373 and xy[1] <= 182:
        if arr[0][0] == "":
            arr[0][0] = chars[turn]
            turn = (turn + 1)%2
            print(arr)
    elif xy[0] <= 583 and xy[1] <= 182:
        if arr[0][1] == "":
            arr[0][1] = chars[turn]
            turn = (turn + 1)%2
            print(arr)
    elif xy[0] > 583 and xy[1] <= 182:
        if arr[0][2] == "":
            arr[0][2] = chars[turn]
            turn = (turn + 1)%2
            print(arr)
    elif xy[0] <= 373 and xy[1] <= 347:
        if arr[1][0] == "":
            arr[1][0] = chars[turn]
            turn = (turn + 1)%2
            print(arr)
    elif xy[0] <= 583 and xy[1] <= 347:
        if arr[1][1] == "":
            arr[1][1] = chars[turn]
            turn = (turn + 1)%2
            print(arr)
    elif xy[0] > 583 and xy[1] <= 347:
        if arr[1][2] == "":
            arr[1][2] = chars[turn]
            turn = (turn + 1)%2
            print(arr)
    elif xy[0] <= 373 and xy[1] > 347:
        if arr[2][0] == "":
            arr[2][0] = chars[turn]
            turn = (turn + 1)%2
            print(arr)
    elif xy[0] <= 583 and xy[1] > 347:
        if arr[2][1] == "":
            arr[2][1] = chars[turn]
            turn = (turn + 1)%2
            print(arr)
    else:
        if arr[2][2] == "":
            arr[2][2] = chars[turn]
            turn = (turn + 1)%2
            print(arr)


def check_winner(grid):
    # Check rows
    for row in grid:
        if row[0] == row[1] == row[2] and row[0] != "":
            print(row[0] + " won")

    # Check columns
    for col in range(3):
        if grid[0][col] == grid[1][col] == grid[2][col] and grid[0][col] != "":
            print(grid[0][col] + " won")

    # Check diagonals
    if grid[0][0] == grid[1][1] == grid[2][2] and grid[0][0] != "":
        print(grid[0][0] + " won")
    if grid[0][2] == grid[1][1] == grid[2][0] and grid[0][2] != "":
        print(grid[0][2] + " won")



def draw_game(indexpos, draw, arr, chars):
    if draw:
        pygame.draw.circle(WIN, (0, 255, 0), indexpos, 7)
    elif indexpos:
        pygame.draw.circle(WIN, (255, 0, 0), indexpos, 7)

    for i in range(len(draws)):
        for j in range(len(draws[i])):
            if j != 0:
                pygame.draw.line(WIN, (255, 0, 255), draws[i][j-1], draws[i][j], 7)
                check_cell(arr, draws[i][j - 1], chars)

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



            if fingers[1] and not fingers[2] and not menu:
                if not drawStart:
                    drawStart = True
                    drawNumber += 1
                    draws.append([])
                draws[drawNumber].append(indexpos)
                draw = True
            else:
                drawStart = False



    if menu:
        WIN.blit(background_image, (0, 0))
        draw_menu(show_text, indexpos)

    else:
        WIN.blit(grid_img, (0, 0))
        draw_game(indexpos, draw, grid_array, chars)
        check_winner(grid_array)

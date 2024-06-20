import pygame
import numpy as np
import mediapipe as mp
import cv2
import math

pygame.init()

# Screen dimensions
GRID_W, GRID_H = 800, 800
ROWS, COLS = 3, 3
CELL_SIZE = GRID_W // COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Screen setup
screen = pygame.display.set_mode((GRID_W, GRID_H))
pygame.display.set_caption('Tic Tac Toe Drawing Recognition')

# Grid
grid = [["" for _ in range(COLS)] for _ in range(ROWS)]
drawings = []
current_drawing = []
startCell = None
# Font for displaying text
font = pygame.font.Font(None, 100)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# State variables
drawing = False

def draw_grid():
    for row in range(1, ROWS):
        pygame.draw.line(screen, BLACK, (0, row * CELL_SIZE), (GRID_W, row * CELL_SIZE), 3)
    for col in range(1, COLS):
        pygame.draw.line(screen, BLACK, (col * CELL_SIZE, 0), (col * CELL_SIZE, GRID_H), 3)

def draw_drawings():
    for drawing in drawings:
        if len(drawing['points']) > 1:
            pygame.draw.lines(screen, BLACK, False, drawing['points'], 3)

def fingers_up(hand_landmarks):
    fingers = []
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[
        mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(True)
    else:
        fingers.append(False)

    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y)
    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y)
    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y)

    return fingers

def get_cell(pos):
    x, y = pos
    row = y // CELL_SIZE
    col = x // CELL_SIZE
    return row, col

def recognize_character(drawing):
    # Placeholder for the model prediction
    # Here you would preprocess the drawing and pass it to the model
    # For example: character = model.predict(preprocess(drawing))
    # Simulating a character recognition for demonstration:
    return "O" if np.random.rand() > 0.5 else "X"

running = True
cap = cv2.VideoCapture(0)
while running:
    screen.fill(WHITE)
    draw_grid()
    draw_drawings()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Game logic
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    res = hands.process(frame)
    indexpos = None
    
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:            
            # Check if the index finger is up and get the position
            fingers = fingers_up(hand_landmarks)
            if fingers[1]:
                ratio_x_to_pixel = lambda x: math.ceil(x * GRID_W)
                ratio_y_to_pixel = lambda y: math.ceil(y * GRID_H)
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                indexpos = (ratio_x_to_pixel(index_tip.x), ratio_y_to_pixel(index_tip.y))
                
                if drawing:
                    indexpos = min(max(indexpos[0], startCell[1] * CELL_SIZE), (startCell[1] + 1) * CELL_SIZE - 1), min(max(indexpos[1], startCell[0] * CELL_SIZE), (startCell[0] + 1) * CELL_SIZE - 1)
                    current_drawing.append(indexpos)
                pygame.draw.circle(screen, (255, 0, 0), indexpos, 7)
                
            if fingers[1] and not fingers[2] and not drawing:
                if not drawing:
                    startCell = get_cell(indexpos)
                drawing = True                
                current_drawing = []
                indexpos = min(max(indexpos[0], startCell[1] * CELL_SIZE), (startCell[1] + 1) * CELL_SIZE - 1), min(max(indexpos[1], startCell[0] * CELL_SIZE), (startCell[0] + 1) * CELL_SIZE - 1)
                current_drawing.append(indexpos)
                
            elif drawing and not (fingers[1] and not fingers[2]):
                drawing = False
                if current_drawing and len(current_drawing) > 1:
                    row, col = get_cell(current_drawing[-1])
                    character = recognize_character(current_drawing)
                    if grid[row][col] == "":
                        grid[row][col] = character
                        drawings.append({'points': current_drawing, 'character': character})
                        current_drawing = []

    # Draw the current drawing if it contains at least 2 points
    if drawing and len(current_drawing) > 1:        
        pygame.draw.lines(screen, BLACK, False, current_drawing, 3)
    
    # Draw the recognized characters on the grid
    for row in range(ROWS):
        for col in range(COLS):
            if grid[row][col] != "":
                text = font.render(grid[row][col], True, BLACK)
                screen.blit(text, (col * CELL_SIZE + CELL_SIZE//3, row * CELL_SIZE + CELL_SIZE//5))
    
    pygame.display.flip()

pygame.quit()

import math
import cv2
import mediapipe as mp
import pygame
import numpy as np
import torch
import torchvision.transforms as transforms
import os

from menu import draw_menu
from game import draw_game
from game import check_winner
from PIL import Image
from model import OurCNN

BASEDIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASEDIR)

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

background_image_path = os.path.join(PARENT_DIR, 'assets', 'images', 'background', 'prova3.jpeg')
background_image = pygame.image.load(background_image_path)
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))
grid_image_path = os.path.join(PARENT_DIR, 'assets', 'images', 'background', 'tic.jpg')
grid_img = pygame.image.load(grid_image_path)
grid_img = pygame.transform.scale(grid_img, (WIDTH, HEIGHT))
MODELPATH = os.path.join(PARENT_DIR, 'models', 'OurCNN2.pth')

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

turn = 0

P_MIN = 0.08


def prob_X_O():
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # x, y, width, height = 192, 53, 199, 167
    boundaries = get_boundaries(startCell, x_coordinates, y_coordinates)
    x, y = boundaries[0], boundaries[2]
    width, height = boundaries[1] - boundaries[0], boundaries[3] - boundaries[2]

    subsurface = WIN.subsurface((x, y, width, height))
    subsurface_array = pygame.surfarray.array3d(subsurface)
    subsurface_array = np.transpose(subsurface_array, (1, 0, 2))
    cropped_image = Image.fromarray(subsurface_array)

    print(startCell)
    model = OurCNN().to(device)
    model.load_state_dict(torch.load(MODELPATH, map_location=torch.device('cpu')))
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),  # Assicurati che l'immagine sia 28x28
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(cropped_image).unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():  # Disabilita il calcolo dei gradienti
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_prob, top_class = probabilities.topk(1, dim=1)
        probabilities = probabilities.squeeze().cpu().numpy()
        prediction = top_class.item()

    letters = [chr(i + 96) for i in range(1, 27)]
    probabilities_dict = {letters[i - 1]: probabilities[i] for i in range(1, 27)}

    x_prob = round(probabilities_dict[letters[23]].item(), 2)
    o_prob = round(probabilities_dict[letters[14]].item(), 2)

    print("\nProbabilità per ogni lettera:")
    print(f'{letters[23]} : {probabilities_dict[letters[23]].item():.2f}')
    print(f'{letters[14]} : {probabilities_dict[letters[14]].item():.2f}')

    if turn == 0:
        if x_prob <= P_MIN and o_prob <= P_MIN:
            return None, None
    elif (turn % 2) == 0:
        if o_prob <= P_MIN:
            return None, None
    else:
        if x_prob <= P_MIN:
            return None, None

    return o_prob, x_prob


def insert_move(grid, cell_index, chars, x_prob, o_prob):
    global turn
    i = cell_index // 3
    j = cell_index % 3
    if grid[i][j] == "":
        if turn == 0:
            if x_prob > o_prob:
                turn = 1
        grid[i][j] = chars[(turn % 2)]
        turn += 1

    print(grid)


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
    return [xs[j], xs[j + 1], ys[i], ys[i + 1]]


def get_cell(xy):
    global turn
    if xy[0] <= 391 and xy[1] <= 220:
        return 0
    elif xy[0] <= 610 and xy[1] <= 220 and xy[0] > 391:
        return 1
    elif xy[0] > 610 and xy[1] <= 220:
        return 2
    elif xy[0] <= 391 and xy[1] <= 420 and xy[1] > 220:
        return 3
    elif xy[0] <= 610 and xy[1] <= 420 and xy[0] > 391 and xy[1] > 220:
        return 4
    elif xy[0] > 610 and xy[1] <= 420 and xy[1] > 220:
        return 5
    elif xy[0] <= 391 and xy[1] > 420:
        return 6
    elif xy[0] <= 610 and xy[1] > 420:
        return 7
    else:
        return 8


def isOccupied(grid, index_pos):
    startCell = get_cell(index_pos)
    i = startCell // 3
    j = startCell % 3
    if grid[i][j] == "":
        return False
    else:
        return True


menu = True
running = True
show_text = True
last_toggle_time = pygame.time.get_ticks()
check_cell = False
Erasing = False
boundaries = None
count = 0

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
            fingers = fingers_up(hand_landmarks)

            # Controllo se l'indice è alzato e prendo la posizione.
            if fingers[1]:
                ratio_x_to_pixel = lambda x: math.ceil(x * WIDTH)
                ratio_y_to_pixel = lambda y: math.ceil(y * HEIGHT)
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # In questo modo posso spostare il pallino verso il basso senza che scompaia o faccia cose strane. (Da sistemare)
                cx = float(np.interp(int(ratio_x_to_pixel(index_tip.x)), [150, WIDTH - 150], [0, WIDTH]))
                cy = float(np.interp(int(ratio_y_to_pixel(index_tip.y)), [150, HEIGHT - 150], [0, HEIGHT]))
                index_pos = (cx, cy)
                if boundaries is not None and count > 0:
                    index_pos = min(max(index_pos[0], boundaries[0]), boundaries[1]), min(
                        max(index_pos[1], boundaries[2]), boundaries[3])

            # Check if the draw is confirmed
            if is_thumb_up_and_fist_closed(hand_landmarks):
                menu = False
                if count > 0:
                    # print("ENTRAAA")
                    check_cell = True
                    count = 0

            if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4] and not fingers[0] and not menu:
                if not isOccupied(grid_array, index_pos):
                    if not drawStart:
                        startCell = get_cell(index_pos)
                        drawStart = True
                        drawNumber += 1
                        draws.append([])
                        count += 1
                    boundaries = get_boundaries(startCell, x_coordinates, y_coordinates)
                    index_pos = min(max(index_pos[0], boundaries[0]), boundaries[1]), min(
                        max(index_pos[1], boundaries[2]), boundaries[3])
                    draws[drawNumber].append(index_pos)
                    draw = True
            else:
                drawStart = False

            if fingers == [False, True, True, True, False]:
                if draws:
                    if drawNumber >= 0:
                        if not Erasing:
                            draws.pop()
                            drawNumber -= 1
                            Erasing = True
                            count -= 1

            else:
                Erasing = False

    if menu:
        WIN.blit(background_image, (0, 0))
        # Disegno punto indice
        if index_pos:
            pygame.draw.circle(WIN, (255, 0, 0), index_pos, 6)
        draw_menu(WIN, show_text, WIDTH, HEIGHT)

    else:

        # ------------------------- Prova -------------------------
        x_prob = None
        o_prob = None

        if check_cell:
            o_prob, x_prob = prob_X_O()
            if o_prob is not None:
                insert_move(grid_array, startCell, chars, x_prob, o_prob)
            else:
                print("RIFAI LA MOSSA")
                # Cacella disegno
            check_cell = False

        WIN.blit(grid_img, (0, 0))
        draw_game(WIN, index_pos, draw, draws, count, turn, x_prob, o_prob)
        check_winner(grid_array)



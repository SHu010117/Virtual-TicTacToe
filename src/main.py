import math
import cv2
import mediapipe as mp
import pygame
import numpy as np
import torch
import torchvision.transforms as transforms
import os
from menu import draw_menu
from game import draw_game, draw_confirm_window
from game import check_winner
from PIL import Image
from model import OurCNN

'''
Here we implement the main file which contains the most important and
basic functions such as:
    - mediapipe functions to detect user's hand
    - gesture recognitions 
    - pygame window initialization
    - letter recognition through our model
    - other basic funtions for the game logic of tic tac toe
'''

# setting path to access to different images
BASEDIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASEDIR)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Error constant used to check hand positions
EPSILON = 0
# Pygame initialization
pygame.init()
pygame.mixer.init()

# Carica e riproduci la musica di sottofondo
# pygame.mixer.music.load('../assets/music/arcade-party-173553.mp3')
# pygame.mixer.music.play(-1)  # loop
# Initializing Window size: Do Not change these parameters!!
WIDTH, HEIGHT = 1000, 680
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe')

# access to the images and resize it to adapt the window size
background_image_path = os.path.join(PARENT_DIR, 'assets', 'images', 'background', 'prova3.jpeg')
background_image = pygame.image.load(background_image_path)
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))
background_image_small = pygame.transform.scale(background_image, (WIDTH // 2, HEIGHT // 2))

grid_image_path = os.path.join(PARENT_DIR, 'assets', 'images', 'background', 'tic.jpg')
grid_img = pygame.image.load(grid_image_path)
grid_img = pygame.transform.scale(grid_img, (WIDTH, HEIGHT))
MODELPATH = os.path.join(PARENT_DIR, 'models', 'OurCNN2.pth')

FPS = 60
CLOCK = pygame.time.Clock()
# webcam initialization
cap = cv2.VideoCapture(0)

# Mediapipe functions to detect hand landmarks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# array which contains the coordinates for drawing
draws = [[]]
drawNumber = 0
drawStart = False
startCell = None

grid_array = [["", "", ""],
              ["", "", ""],
              ["", "", ""]]
chars = ["O", "X"]
turn = 0
P_MIN = 0.08


# verifying which caracter(O/X) has been drawn using the our CNN model
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

    return o_prob, x_prob


# checking which is the first character to be drawn
def first_move(x_prob, o_prob):
    global turn
    if turn == 0 and x_prob > o_prob:
        turn = 1


# register a valid move
def insert_move(grid, cell_index, chars):
    global turn
    i = cell_index // 3
    j = cell_index % 3
    if grid[i][j] == "":
        grid[i][j] = chars[(turn % 2)]
        turn += 1

    print(grid)


# checking which fingers are up
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


# funciton to detect thumb up or down
def is_thumb_up_or_down_and_fist_closed(hand_landmarks):
    # hand tips positions
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # check whether thumb's up or down
    is_thumb_up = thumb_tip.y < thumb_mcp.y
    is_thumb_down = thumb_tip.y > thumb_mcp.y

    # Check if the fist is closed:
    is_fist_closed = (
            (min(thumb_mcp.x, index_mcp.x) - EPSILON <= index_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON) and
            (min(thumb_mcp.x, index_mcp.x) - EPSILON <= middle_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON) and
            (min(thumb_mcp.x, index_mcp.x) - EPSILON <= ring_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON) and
            (min(thumb_mcp.x, index_mcp.x) - EPSILON <= pinky_tip.x <= max(thumb_mcp.x, index_mcp.x) + EPSILON)
    )

    return is_thumb_up and is_fist_closed, is_thumb_down and is_fist_closed


x_coordinates = [192, 391, 610, 784]
y_coordinates = [53, 220, 420, 640]


# boundaries for the given cell index
def get_boundaries(index, xs, ys):
    i = index // 3
    j = index % 3
    return [xs[j], xs[j + 1], ys[i], ys[i + 1]]


# get the cell index
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


def remove_draw(count):
    while count > 0:
        draws.pop()
        count -= 1


def calculate_point(prob):
    return 1 + (((prob - P_MIN) * 99) / (1 - P_MIN))


# other variables used for the correct execution during the loop
menu = True
running = True
show_text = True
last_toggle_time = pygame.time.get_ticks()
check_cell = False
Erasing = False
boundaries = None
count = 0
x_prob = None
o_prob = None
confirm_window = False
puntX = 0
puntO = 0
match_done = False
PIXELPATH = os.path.join(PARENT_DIR, 'assets', 'fonts', 'public-pixel-font', 'PublicPixel-E447g.ttf')
winner = None
winning_cells = None
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

    # get the webcam frame
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    res = hands.process(frame)
    index_pos = None
    draw = False
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            fingers = fingers_up(hand_landmarks)
            thumb_up, thumb_down = is_thumb_up_or_down_and_fist_closed(hand_landmarks)

            # picking index tip's position when it's up
            if fingers[1]:
                ratio_x_to_pixel = lambda x: math.ceil(x * WIDTH)
                ratio_y_to_pixel = lambda y: math.ceil(y * HEIGHT)
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cx = float(np.interp(int(ratio_x_to_pixel(index_tip.x)), [150, WIDTH - 150], [0, WIDTH]))
                cy = float(np.interp(int(ratio_y_to_pixel(index_tip.y)), [150, HEIGHT - 150], [0, HEIGHT]))
                index_pos = (cx, cy)
                # limiting the movement of user's hand in the cell when the started drawing
                if boundaries is not None and count > 0:
                    index_pos = min(max(index_pos[0], boundaries[0]), boundaries[1]), min(
                        max(index_pos[1], boundaries[2]), boundaries[3])

            # thumb up to start the game in the menu, or confirm the drawing during the game
            if thumb_up and not match_done:
                menu = False
                if count > 0:
                    check_cell = True

            # open hand to confirm when deciding to return to menu window
            if fingers == [True, True, True, True, True] and confirm_window:
                menu = True

            # fist closed to decline return to window
            if fingers == [False, False, False, False, False] and confirm_window:
                confirm_window = False

            # when only index finger's up, start drawing
            if fingers == [False, True, False, False, False] and not menu and not match_done and not confirm_window:
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
                    # print(drawNumber)
                    draws[drawNumber].append(index_pos)
                    draw = True
            else:
                drawStart = False

            # gesture to delete last segment
            if fingers == [False, True, True, True, False]:
                if draws:
                    if drawNumber >= 0 and count > 0:
                        if not Erasing:
                            draws.pop()
                            drawNumber -= 1
                            Erasing = True
                            count -= 1
            else:
                Erasing = False

            # gesture to start a new game when the game's done
            if (fingers == [False, True, False, False, True] and match_done) or (
                    fingers == [True, True, True, True, True] and confirm_window):
                match_done = False
                draws = [[]]
                drawNumber = 0
                # startCell = None
                grid_array = [["", "", ""], ["", "", ""], ["", "", ""]]
                puntO = 0
                puntX = 0
                count = 0
                o_prob = None
                x_prob = None
                confirm_window = False
                winner = None
                winning_cells = None
                turn = 0

            # Se vuoi uscire nel menù
            if thumb_down and not menu:
                confirm_window = True

    if menu:
        # show menu window
        WIN.blit(background_image, (0, 0))
        # Disegno punto indice
        if index_pos:
            pygame.draw.circle(WIN, (255, 0, 0), index_pos, 6)
        draw_menu(WIN, show_text, WIDTH, HEIGHT)
    else:
        # once the user confirmed their drawing, the cell is sent to model to verify the character
        if check_cell:
            o_prob, x_prob = prob_X_O()
            first_move(x_prob, o_prob)
            if (o_prob > P_MIN and (turn % 2) == 0):
                # print(o_prob)
                ris = calculate_point(o_prob)
                if puntO != 0:
                    puntO += ris
                    puntO /= 2
                else:
                    puntO += ris

                puntO = round(puntO, 2)
                insert_move(grid_array, startCell, chars)
                winner, winning_cells = check_winner(grid_array)
                x_prob = None
                o_prob = None
            elif (x_prob > P_MIN and (turn % 2) == 1):
                # un po' uno schifo, ma funziona :D
                ris = calculate_point(x_prob)
                if puntX != 0:
                    puntX += ris
                    puntX /= 2
                else:
                    puntX += ris
                puntX = round(puntX, 2)
                insert_move(grid_array, startCell, chars)
                winner, winning_cells = check_winner(grid_array)

                x_prob = None
                o_prob = None
            else:
                remove_draw(count)
                drawNumber -= count
            count = 0
            check_cell = False

        # show game window
        WIN.blit(grid_img, (0, 0))

        draw_game(WIN, index_pos, draw, draws, count, turn, x_prob, o_prob, P_MIN, puntX, puntO, winning_cells, winner,
                  match_done)
        if confirm_window:
            draw_confirm_window(WIN, WIDTH, HEIGHT, background_image_small)
        if winner:
            match_done = True
        pygame.display.flip()
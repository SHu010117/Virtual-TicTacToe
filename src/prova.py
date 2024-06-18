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

    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_PIP].y)
    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.RING_FINGER_PIP].y)
    fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[
        mp_hands.HandLandmark.PINKY_PIP].y)

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
                    current_drawing.append(indexpos)
                pygame.draw.circle(screen, (255, 0, 0), indexpos, 7)

            if fingers[1] and not fingers[2] and not drawing:
                drawing = True
                current_drawing = []
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
                screen.blit(text, (col * CELL_SIZE + CELL_SIZE // 3, row * CELL_SIZE + CELL_SIZE // 5))

    pygame.display.flip()

pygame.quit()
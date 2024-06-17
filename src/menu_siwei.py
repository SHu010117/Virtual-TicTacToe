import cv2
import mediapipe as mp
import pygame
import os
import numpy as np


parent_dir = "C:/Users/Siwei Hu/Desktop/AILab/Virtual-TicTacToe"
# Paths to fonts (ensure these paths are correct)
SUPERFTIMEPATH = parent_dir + '/assets/fonts/super-funtime-font/SuperFuntime-3zpLX.ttf'
THEGLOBEFONTPATH = parent_dir + '/assets/fonts/the-globe-font/TheGlobePersonalUseBold-2ORlw.ttf'
NINTENDOPATH = parent_dir + '/assets/fonts/ro-spritendo-font/RoSpritendoSemiboldBeta-vmVwZ.otf'
PIXELPATH = parent_dir + '/assets/fonts/public-pixel-font/PublicPixel-E447g.ttf'

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DGREEN = (24, 60, 37)
DGREEN2 = (24, 60, 37)

# Initialization of Pygame
pygame.init()
pygame.mixer.init()

# Load and play background music (uncomment if needed)
# pygame.mixer.music.load('../assets/music/arcade-party-173553.mp3')
# pygame.mixer.music.play(-1)  # loop

# Create Window
WIDTH, HEIGHT = 1000, 680
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe')

# Check if the background image file exists before loading
background_image_path = parent_dir + '/assets/images/prova3.jpeg'

# Load the background image
background_image = pygame.image.load(background_image_path)
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))



FPS = 30
CLOCK = pygame.time.Clock()


# Initialization of Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
def draw_menu():
    WIN.blit(background_image, (0, 0))

    # Main title
    font = pygame.font.Font(NINTENDOPATH, 70)
    text = font.render('TIC-TAC-TOE', True, DGREEN)
    WIN.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 8))

    # Subtitle
    font = pygame.font.Font(NINTENDOPATH, 29)
    text = font.render('Pollice in sù per iniziare', True, DGREEN2)
    WIN.blit(text, (WIDTH // 2 - text.get_width() // 2, (HEIGHT // 4) + 65))

    # Update the entire screen
    pygame.display.flip()

def fingers_up(mp_hands, hand_landmarks):
        """
        Determine which fingers are up.
        
        Arguments:
        hand_landmarks -- list of hand landmarks

        Returns:
        list of booleans representing whether each finger is up (True) or down (False)
        """
        fingers = []

        # Thumb (landmark 4 is the tip, landmark 3 is the knuckle)
        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
            fingers.append(True)
        else:
            fingers.append(False)

        # Fingers (landmark 8 is the tip, landmark 6 is the pip joint)
        fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y)
        fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
        fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y)
        fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y)

        return fingers

menu = True
running = True

# Initialization of cv2
cap = cv2.VideoCapture(0)
# Main loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Game logic
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    res = hands.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                fingers = fingers_up(mp_hands, hand)
                
                
                if fingers == [True, False, False, False, False]: # thumb's up
                    # Do something
                    print("ok")
                    menu = False
                if fingers == [False, False, False, False, True]: # mignolo è alzato
                    # Do something
                    pass
    if menu:
        WIN.blit(background_image, (0, 0))
        draw_menu()
        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                fingers = fingers_up(mp_hands, hand)
                
                
                if fingers == [True, False, False, False, False]: # thumb's up
                    # Do something
                    print("ok")
                    menu = False
                if fingers == [False, False, False, False, True]: # mignolo è alzato
                    # Do something
                    pass
                
    else:
        WIN.fill(WHITE)  # This is a placeholder for other game states

    # Control frame rate
    CLOCK.tick(FPS)

# Release resources
cap.release()
pygame.quit()



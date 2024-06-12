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
DGREEN = (24,60,37)
DGREEN2 = (24,60,37)



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


FPS = 60
CLOCK = pygame.time.Clock()

# Inizializzazione cv2
cap = cv2.VideoCapture(0)

# Inizializzazione di Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils



def draw_menu():
    # WIN.fill((245,255,255))

    # Scritta principale
    font = pygame.font.Font(PIXELPATH, 78)
    text = font.render('TIC-TAC-TOE', True, DGREEN)
    WIN.blit(text, (WIDTH//2 - text.get_width()//2, (HEIGHT//8)))

    # Scritta secondaria
    font = pygame.font.Font(PIXELPATH, 24)
    text = font.render('Pollice in sù per iniziare', True, DGREEN2)
    WIN.blit(text, (WIDTH // 2 - text.get_width() // 2, (HEIGHT // 4) + 65))

    # Aggiorna l'intera schermata
    pygame.display.flip()






# TODO: implmentare funzione che riconosca il pollice alzato



# TODO: implementare funzione che riconosca un segno (da decidere) per abbassare la musica di sottofondo
# TODO: implementare funzione che riconosca un segno (da decidere) per uscire dal gioco.


menu = True
running = True

# Main loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()

    # Logica del gioco
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(frame)
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
        draw_menu()
        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                print("dajeroma")




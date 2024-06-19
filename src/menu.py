from urllib.parse import quote

import pygame
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PARENT_DIR = os.path.dirname(BASE_DIR)
PIXELPATH = os.path.join(PARENT_DIR, 'assets', 'fonts', 'public-pixel-font', 'PublicPixel-E447g.ttf')


DARK_GREEN = (24, 60, 37)
DARK_GREEN2 = (24, 60, 37)


def draw_menu(win, show_text, width, height):
    # WIN.fill((245,255,255))

    # Scritta principale
    font = pygame.font.Font(PIXELPATH, 78)
    text = font.render('TIC-TAC-TOE', True, DARK_GREEN)
    win.blit(text, (width // 2 - text.get_width() // 2, (height // 8)))

    # Scritta secondaria
    if show_text:
        font = pygame.font.Font(PIXELPATH, 24)
        text = font.render('Pollice in sù per iniziare', True, DARK_GREEN2)
        win.blit(text, (width // 2 - text.get_width() // 2, (height // 4) + 65))

    # Aggiorna l'intera schermata
    pygame.display.flip()

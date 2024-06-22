import pygame
import os

'''
Here's the menu interface
'''

BASEDIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASEDIR)
PIXELPATH = os.path.join(PARENT_DIR, 'assets', 'fonts', 'public-pixel-font', 'PublicPixel-E447g.ttf')
DARK_GREEN = (24, 60, 37)
DARK_GREEN2 = (24, 60, 37)


def draw_menu(win, show_text, width, height):
    
    '''
    We build the munu page through pygame library
    '''

    # Main message
    font = pygame.font.Font(PIXELPATH, 78)
    text = font.render('TIC-TAC-TOE', True, DARK_GREEN)
    win.blit(text, (width // 2 - text.get_width() // 2, (height // 8)))

    # Secondary message
    if show_text:
        font = pygame.font.Font(PIXELPATH, 24)
        text = font.render('Pollice in sù per iniziare', True, DARK_GREEN2)
        win.blit(text, (width // 2 - text.get_width() // 2, (height // 4) + 65))

    # refresh the window
    pygame.display.flip()

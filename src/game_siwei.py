import math
import pygame
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASEDIR)
intersection_points = [(373, 182), (583, 182), (373, 347), (583, 347)]
turn = 0
PINKYUP = "C:/Users/Siwei Hu/Desktop/AI-Lab/Virtual-TicTacToe/assets/images/game images/pinky_up.png"
pinky_up_img = pygame.image.load(PINKYUP)
pinky_up_img = pygame.transform.scale(pinky_up_img, (50, 50))

DRAWPATH = os.path.join(PARENT_DIR, 'assets', 'images', 'game images', 'pointing-right_237663.png')
draw_icon = pygame.image.load(DRAWPATH)
draw_icon = pygame.transform.scale(draw_icon, (55, 55))
draw_icon = pygame.transform.rotate(draw_icon, 90)


PIXELPATH = os.path.join(PARENT_DIR, 'assets', 'fonts', 'public-pixel-font', 'PublicPixel-E447g.ttf')
WHITE = (255, 255, 255)

ICONPATH = "C:/Users/Siwei Hu/Desktop/AI-Lab/Virtual-TicTacToe/assets/images/game images/two_fing.png"
move_icon = pygame.image.load(ICONPATH)
move_icon = pygame.transform.scale(move_icon, (50, 50))


def insert_move(grid, cell_index, chars):
    global turn
    i = cell_index//3
    j = cell_index%3
    if grid[i][j] == "":
        grid[i][j] = chars[turn]
        turn = (turn + 1) % 2 
        print(grid)

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


#def draw_game(win, index_pos, draw, arr, chars, draws):
def draw_game(win, index_pos, draw, grid, chars, index_cell, draws):
    # Immagine mignolo
    win.blit(pinky_up_img, (10, 10))
    # testo vicino
    font = pygame.font.Font(PIXELPATH, 16)
    text = font.render('MENU', True, WHITE)
    win.blit(text, (10, 70))

    win.blit(draw_icon, (10, 100))
    text = font.render('DRAW', True, WHITE)
    win.blit(text, (10, 160))

    win.blit(move_icon, (10, 190))
    text = font.render('MOVE', True, WHITE)
    win.blit(text, (10, 250))

    if draw:
        pygame.draw.circle(win, (0, 255, 0), index_pos, 7)
        # print(index_pos)

    elif index_pos:
        pygame.draw.circle(win, (255, 0, 0), index_pos, 7)

    for i in range(len(draws)):
        for j in range(len(draws[i])):
            if j != 0:
                pygame.draw.line(win, (255, 255, 255), draws[i][j-1], draws[i][j], 7)
                insert_move(grid, index_cell, chars)

    pygame.display.flip()
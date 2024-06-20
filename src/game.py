import math
import pygame
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASEDIR)
intersection_points = [(373, 182), (583, 182), (373, 347), (583, 347)]
turn = 0
PINKYUPPATH = os.path.join(PARENT_DIR, 'assets', 'images', 'game images', 'pinky_up.png')
pinky_up_img = pygame.image.load(PINKYUPPATH)
pinky_up_img = pygame.transform.scale(pinky_up_img, (50, 50))

PUGNOPATH = os.path.join(PARENT_DIR, 'assets', 'images', 'game images', 'pugno.png')
fist_img = pygame.image.load(PUGNOPATH)
fist_img = pygame.transform.scale(fist_img, (50, 50))

APERTAPATH = os.path.join(PARENT_DIR, 'assets', 'images', 'game images', 'manoaperta.png')
open_img = pygame.image.load(APERTAPATH)
open_img = pygame.transform.scale(open_img, (50, 50))

DRAWPATH = os.path.join(PARENT_DIR, 'assets', 'images', 'game images', 'pointing-right_237663.png')
draw_icon = pygame.image.load(DRAWPATH)
draw_icon = pygame.transform.scale(draw_icon, (55, 55))
draw_icon = pygame.transform.rotate(draw_icon, 90)

PIXELPATH = os.path.join(PARENT_DIR, 'assets', 'fonts', 'public-pixel-font', 'PublicPixel-E447g.ttf')
WHITE = (255, 255, 255)
NICE_RED = (255, 49, 49)

ICONPATH = os.path.join(PARENT_DIR, 'assets', 'images', 'game images', 'two_fing.png')
move_icon = pygame.image.load(ICONPATH)
move_icon = pygame.transform.scale(move_icon, (50, 50))

THUMBUP = os.path.join(PARENT_DIR, 'assets', 'images', 'game images', 'thumbs-up.png')
confirm_icon = pygame.image.load(THUMBUP)
confirm_icon = pygame.transform.scale(confirm_icon, (90, 90))
confirm_icon = pygame.transform.flip(confirm_icon, True, False)

ROCKNROLLPATH = os.path.join(PARENT_DIR, 'assets', 'images', 'game images', 'Rock.png')
newgame_icon = pygame.image.load(ICONPATH)
newgame_icon = pygame.transform.scale(move_icon, (50, 50))


DARK_GREEN = (24, 60, 37)

x_coordinates = [192, 391, 610, 784]
y_coordinates = [53, 220, 420, 640]


def check_winner(grid):
    # Check rows
    for row in range(3):
        if grid[row][0] == grid[row][1] == grid[row][2] and grid[row][0] != "":
            return grid[row][0], row

    # Check columns
    for col in range(3):
        if grid[0][col] == grid[1][col] == grid[2][col] and grid[0][col] != "":
            return grid[0][col], col + 3

    # Check diagonals
    if grid[0][0] == grid[1][1] == grid[2][2] and grid[0][0] != "":
        return grid[0][0], 6
    if grid[0][2] == grid[1][1] == grid[2][0] and grid[0][2] != "":
        return grid[0][2], 7
    count = 0
    for row in range(3):
        for col in range(3):
            if grid[row][col] != "":
                count += 1
    if count == 9:
        return "Pareggio", 8
    return None, []


'''
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
'''


def find_points(cell_index, xs, ys):
    if cell_index < 3:
        y = ys[cell_index] + (ys[cell_index + 1] - ys[cell_index]) / 2
        start_x = xs[0]
        end_x = xs[-1]
        return (start_x, y), (end_x, y)
    elif cell_index < 6:
        cell_index -= 3
        x = xs[cell_index] + (xs[cell_index + 1] - xs[cell_index]) / 2
        start_y = ys[0]
        end_y = ys[-1]
        return (x, start_y), (x, end_y)
    elif cell_index == 6:
        return (xs[0], ys[0]), (xs[-1], ys[-1])
    return (xs[-1], ys[0]), (xs[0], ys[-1])


def draw_winner_line(win, cells, winner, po, px):
    start_pos, end_pos = find_points(cells, x_coordinates, y_coordinates)
    if winner != "Pareggio":
        pygame.draw.line(win, (255, 255, 255), start_pos, end_pos, 5)
    font = pygame.font.Font(PIXELPATH, 20)
    if winner == "Pareggio":
        if po > px:
            text = "Pareggio: O ha fatto più punti"
        elif px > po:
            text = "Pareggio: X ha fatto più punti"
        else:
            text = "Pareggio: stesso punteggio"
    elif winner == "O":
        text = "O ha vinto facendo tris"
    else:
        text = "X ha vinto facendo tris"

    result = font.render(text, True, (255, 49, 49))
    win.blit(result, (200, 300))
    # pygame.display.flip()

def draw_confirm_window(win,width,height, bg):
    win.blit(bg, (width // 2 - (width // 4), (height // 4)))

    font = pygame.font.Font(PIXELPATH, 30)
    text = font.render('CONFERMA', True, DARK_GREEN)
    win.blit(text, ((width // 2 - (width // 4) + 120), (height // 4) + 40))

    win.blit(open_img, ((width // 2 - (width // 4) + text.get_width() + 140), (height // 4) + 30))


    text = font.render('CONTINUA', True, DARK_GREEN)
    win.blit(text, ((width // 2 - (width // 4) + 120), (height // 4) + 140))
    win.blit(fist_img, ((width // 2 - (width // 4) + text.get_width() + 140), (height // 4) + 130))


    # pygame.display.flip()

#def draw_game(win, index_pos, draw, arr, chars, draws, count):
def draw_game(win, index_pos, draw, draws, count, turn, x_prob, o_prob, p_min, puntX, puntO, winning_cells, winner, match_done):
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

    font = pygame.font.Font(PIXELPATH, 11)

    text = font.render(f'AVERAGE X : {puntX}', True, WHITE)
    win.blit(text, (8, 580))

    text = font.render(f'AVERAGE O : {puntO}', True, WHITE)
    win.blit(text, (8, 610))

    if draw:
        pygame.draw.circle(win, (0, 255, 0), index_pos, 7)
        # print(index_pos)
    elif index_pos:
        pygame.draw.circle(win, (255, 0, 0), index_pos, 7)
        # print(index_pos)

    if count > 0:
        win.blit(confirm_icon, (855, 307))
        text = font.render('CONFIRM', True, WHITE)
        win.blit(text, (850, 404))

    if o_prob is not None:
        if ((turn % 2) == 0 and o_prob <= p_min) or ((turn % 2) == 1 and x_prob <= p_min):
            # print("entri")
            font = pygame.font.Font(PIXELPATH, 20)
            text = font.render('Your drawing is very bad. Try again.', True, NICE_RED)
            win.blit(text, (170, 20))
    if match_done:
        win.blit(newgame_icon, (855, 307))
        text = font.render('New Game', True, WHITE)
        win.blit(text, (850, 404))


    for i in range(len(draws)):
        for j in range(len(draws[i])):
            if j != 0:
                pygame.draw.line(win, (255, 255, 255), draws[i][j - 1], draws[i][j], 7)

    if winner:
        draw_winner_line(win, winning_cells, winner, puntO, puntX)


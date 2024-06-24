import pygame
import os

'''
This is the file which allows the visualization of the game window through pygame
Different images are imported which are icons of hand gestures to inform the users
 which gesture to adopt for the desired action/move. Also the drawings are made 
 visible through implemented functions
'''

# different icon images for the hand gestures
BASEDIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASEDIR)

def load_and_scale(PARENT_DIR, relative_path):
    PATH = os.path.join(PARENT_DIR, 'assets', 'images', 'game images', relative_path)
    icon = pygame.image.load(PATH)
    icon = pygame.transform.scale(icon, (50, 50))
    return icon

pinky_up_img = load_and_scale(PARENT_DIR, 'pinky_up.png')
fist_img = load_and_scale(PARENT_DIR, 'pugno.png')
open_img = load_and_scale(PARENT_DIR, 'manoaperta.png')
delete_img = load_and_scale(PARENT_DIR, 'tap.png')
delete_img = pygame.transform.scale(delete_img, (90, 90))
draw_icon = load_and_scale(PARENT_DIR, 'pointing-right_237663.png')
draw_icon = pygame.transform.rotate(draw_icon, 90)
PIXELPATH = os.path.join(PARENT_DIR, 'assets', 'fonts', 'public-pixel-font', 'PublicPixel-E447g.ttf')
WHITE = (255, 255, 255)
NICE_RED = (255, 49, 49)
move_icon = load_and_scale(PARENT_DIR, 'two_fing.png')
confirm_icon = load_and_scale(PARENT_DIR, 'thumbs-up.png')
confirm_icon = pygame.transform.scale(confirm_icon, (90, 90))
confirm_icon = pygame.transform.flip(confirm_icon, True, False)
menu_icon = load_and_scale(PARENT_DIR, 'pollice.png')
menu_icon = pygame.transform.flip(menu_icon, True, False)
menu_icon = pygame.transform.rotate(menu_icon, 270)
newgame_icon = load_and_scale(PARENT_DIR, 'Rock.png')
DARK_GREEN = (24, 60, 37)
RED_ORANGE = (255, 68, 51)
x_coordinates = [192, 391, 610, 784]
y_coordinates = [53, 220, 420, 640]

# checking win condition
def check_winner(grid):
    
    '''
    Function which detects the winning condition by analizing the rid array
     and it's called each time a valid move is completed
    It returns the corresponding winner (character O/X or "Tie), and a interger
     which tells the winning combo: 0-1-2 for the rows, 3-4-5 for the columns
     and 6-8 the diagonals. 8 for the tie result
    '''
    
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
        return "Tie", 8
    return None, []

# finds the coordinates for line which acrosses the winnin cells
def find_points(result_int, xs, ys):
    
    '''
    Given the integer which represent a result and arrays of coordinates
     of the grid, this function finds the coordinate of the line which 
     acrosses the grid cells of winning combination
    '''
    
    if result_int < 3:
        y = ys[result_int] + (ys[result_int + 1] - ys[result_int]) / 2
        start_x = xs[0]
        end_x = xs[-1]
        return (start_x, y), (end_x, y)
    elif result_int < 6:
        result_int -= 3
        x = xs[result_int] + (xs[result_int + 1] - xs[result_int]) / 2
        start_y = ys[0]
        end_y = ys[-1]
        return (x, start_y), (x, end_y)
    elif result_int == 6:
        return (xs[0], ys[0]), (xs[-1], ys[-1])
    return (xs[-1], ys[0]), (xs[0], ys[-1])

# draw the line of victory
def draw_winner_line(win, result_int, winner, po, px):
    
    '''
    After detecting the result of game, we find first the
     coordinates of line and draw it on the game window,
     the we show the result message on the screen
    '''
    
    start_pos, end_pos = find_points(result_int, x_coordinates, y_coordinates)
    if winner != "Tie":
        pygame.draw.line(win, (0,255,0), start_pos, end_pos, 7)
    font = pygame.font.Font(PIXELPATH, 25)
    if winner == "Tie":
        if po > px:
            text = "Tied game: O has more points"
        elif px > po:
            text = "Tied game: X has more points"
        else:
            text = "Tied game: same points for both"
    elif winner == "O":
        text = "Game ended: O has won this game!"
    else:
        text = "Game ended: X has won this game!"

    result = font.render(text, True, RED_ORANGE)
    win.blit(result, (185, 300))

# confirm window when trying to return to menu
def draw_confirm_window(win,width,height, bg):
    '''
    We implemented a confirm window which pops out
     when the user wishes to return back to the menu
     page.
    '''
    win.blit(bg, (width // 2 - (width // 4), (height // 4)))

    font = pygame.font.Font(PIXELPATH, 30)
    text = font.render('CONFIRM', True, DARK_GREEN)
    win.blit(text, ((width // 2 - (width // 4) + 120), (height // 4) + 40))

    win.blit(open_img, ((width // 2 - (width // 4) + text.get_width() + 140), (height // 4) + 30))


    text = font.render('CONTINUE', True, DARK_GREEN)
    win.blit(text, ((width // 2 - (width // 4) + 120), (height // 4) + 140))
    win.blit(fist_img, ((width // 2 - (width // 4) + text.get_width() + 140), (height // 4) + 130))

# Function which allows user to see their drawing on the grid 
def draw_game(win, index_pos, draw, draws, count, turn, x_prob, o_prob, p_min, puntX, puntO, winning_cells, winner, match_done):
    
    '''
    Here's the key function which makes visible the following:
        - gestiture icons
        - user's drawings
        - a point which allows user to keep track of their index finger
        - warning messages
        - drawing points
    '''
    
    win.blit(menu_icon, (10, 10))
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

    # red dot for index finger tip:
    if draw:
        pygame.draw.circle(win, (0, 255, 0), index_pos, 7)
        # print(index_pos)
    elif index_pos:
        pygame.draw.circle(win, (255, 0, 0), index_pos, 7)
        # print(index_pos)

    # when user made a drawing we show icon of confirm or delete gesture
    if count > 0:
        win.blit(confirm_icon, (855, 187))
        text = font.render('CONFIRM', True, WHITE)
        win.blit(text, (870, 284))

        win.blit(delete_img, (855, 350))
        text = font.render('DELETE', True, WHITE)
        win.blit(text, (870, 447))

    # warning message when a bad drawing or wrong move is verified
    if o_prob is not None:
        if((turn % 2) == 1 and o_prob >= p_min) or ((turn % 2) == 0 and x_prob >= p_min):
            font = pygame.font.Font(PIXELPATH, 20)
            text = font.render("It is not you turn. Try again.", True, NICE_RED)
            win.blit(text, (170, 20))
        elif ((turn % 2) == 0 and o_prob <= p_min) or ((turn % 2) == 1 and x_prob <= p_min):
            font = pygame.font.Font(PIXELPATH, 20)
            text = font.render('Your drawing is very bad. Try again.', True, NICE_RED)
            win.blit(text, (170, 20))
    
    # icon of new game gesture
    if match_done:
        font = pygame.font.Font(PIXELPATH, 16)
        win.blit(newgame_icon, (875, 320))
        text = font.render('New Game', True, WHITE)
        win.blit(text, (840, 390))

    # user's drawings
    for i in range(len(draws)):
        for j in range(len(draws[i])):
            if j != 0:
                pygame.draw.line(win, (255, 255, 255), draws[i][j - 1], draws[i][j], 7)

    # if match is over
    if winner:
        draw_winner_line(win, winning_cells, winner, puntO, puntX)


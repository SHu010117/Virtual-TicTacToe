import math

import pygame

intersection_points = [(391, 182), (610, 182), (391, 420), (610, 420)]
turn = 0


def print_game(arr):
    for i in range(len(arr)):
        print(arr[i])


def check_cell(arr, xy, chars):
    global turn
    if xy[0] <= 391 and xy[1] <= 182:
        if arr[0][0] == "":
            arr[0][0] = chars[turn]
            turn = (turn + 1) % 2
            print_game(arr)
    elif xy[0] <= 610 and xy[1] <= 182:
        if arr[0][1] == "":
            arr[0][1] = chars[turn]
            turn = (turn + 1) % 2
            print(f"----- TURNO GIOCATORE {turn} -----")
            print_game(arr)

    elif xy[0] > 610 and xy[1] <= 182:
        if arr[0][2] == "":
            arr[0][2] = chars[turn]
            turn = (turn + 1) % 2
            print(f"----- TURNO GIOCATORE {turn} -----")
            print_game(arr)

    elif xy[0] <= 391 and xy[1] <= 420:
        if arr[1][0] == "":
            arr[1][0] = chars[turn]
            turn = (turn + 1) % 2
            print(f"----- TURNO GIOCATORE {turn} -----")
            print_game(arr)

    elif xy[0] <= 610 and xy[1] <= 420:
        if arr[1][1] == "":
            arr[1][1] = chars[turn]
            turn = (turn + 1) % 2
            print(f"----- TURNO GIOCATORE {turn} -----")
            print_game(arr)

    elif xy[0] > 610 and xy[1] <= 420:
        if arr[1][2] == "":
            arr[1][2] = chars[turn]
            turn = (turn + 1) % 2
            print(f"----- TURNO GIOCATORE {turn} -----")
            print_game(arr)

    elif xy[0] <= 391 and xy[1] > 420:
        if arr[2][0] == "":
            arr[2][0] = chars[turn]
            turn = (turn + 1) % 2
            print(f"----- TURNO GIOCATORE {turn} -----")
            print_game(arr)

    elif xy[0] <= 610 and xy[1] > 420:
        if arr[2][1] == "":
            arr[2][1] = chars[turn]
            turn = (turn + 1) % 2
            print(f"----- TURNO GIOCATORE {turn} -----")
            print_game(arr)

    else:
        if arr[2][2] == "":
            arr[2][2] = chars[turn]
            turn = (turn + 1) % 2
            print(f"----- TURNO GIOCATORE {turn} -----")
            print_game(arr)


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


def draw_game(win, index_pos, draw, arr, chars, draws):
    if draw:
        pygame.draw.circle(win, (0, 255, 0), index_pos, 7)
    elif index_pos:
        pygame.draw.circle(win, (255, 0, 0), index_pos, 7)

    for i in range(len(draws)):
        for j in range(len(draws[i])):
            if j != 0:
                pygame.draw.line(win, (255, 0, 255), draws[i][j - 1], draws[i][j], 7)
                check_cell(arr, draws[i][j - 1], chars)

    pygame.display.flip()

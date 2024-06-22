import os
import cv2
import pygame
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
from model import OurCNN
from menu import draw_menu
from game import draw_game, draw_confirm_window, check_winner
import constants

class TicTacToeGame:
    def __init__(self):
        self.BASEDIR = os.path.dirname(os.path.abspath(__file__))
        self.PARENT_DIR = os.path.dirname(self.BASEDIR)

        self.WHITE = (255, 255, 255)

        pygame.init()
        pygame.mixer.init()

        self.WIDTH, self.HEIGHT = constants.WIDTH, constants.HEIGHT
        self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Tic Tac Toe')

        '''
        self.background_image = self.load_and_scale_image('assets/images/background/prova3.jpeg', self.WIDTH, self.HEIGHT)
        self.background_image_small = pygame.transform.scale(self.background_image, (self.WIDTH // 2, self.HEIGHT // 2))
        self.grid_img = self.load_and_scale_image('assets/images/background/tic.jpg', self.WIDTH, self.HEIGHT)
        '''
        self.background_image_path = os.path.join(self.PARENT_DIR, 'assets', 'images', 'background', 'prova3.jpeg')
        self.background_image = pygame.image.load(self.background_image_path)
        self.background_image = pygame.transform.scale(self.background_image, (self.WIDTH, self.HEIGHT))
        self.background_image_small = pygame.transform.scale(self.background_image, (self.WIDTH // 2, self.HEIGHT // 2))

        self.grid_image_path = os.path.join(self.PARENT_DIR, 'assets', 'images', 'background', 'tic.jpg')
        self.grid_img = pygame.image.load(self.grid_image_path)
        self.grid_img = pygame.transform.scale(self.grid_img, (self.WIDTH, self.HEIGHT))
        
        self.MODELPATH = os.path.join(self.PARENT_DIR, 'models', 'OurCNN2.pth')
        self.PIXELPATH = os.path.join(self.PARENT_DIR, 'assets', 'fonts', 'public-pixel-font', 'PublicPixel-E447g.ttf')

        self.FPS = 60
        self.CLOCK = pygame.time.Clock()

        self.puntO = 0
        self.puntX = 0
        self.x_prob = None
        self.y_prob = None
        self.turn = 0
        self.draws = [[]]
        self.drawNumber = 0
        self.drawStart = False
        self.startCell = None
        self.count = 0
        self.grid_array = [["" for _ in range(3)] for _ in range(3)]
        self.chars = ["O", "X"]
        self.boundaries = None
        self.confirm_window = False
        self.menu = True
        self.check_cell = False
        self.match_done = False
        self.Erasing = False
        self.index_pos = None
        self.P_MIN = 0.08
        self.x_coordinates = [192, 391, 610, 784]
        self.y_coordinates = [53, 220, 420, 640]
        self.match_done = False
        self.winner = None
        self.winning_cells = None
        self.draw = False

    def run(self, hand_tracker):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    cv2.destroyAllWindows()
                    hand_tracker.release()
                    exit()
            self.draw = False
            ret, frame = hand_tracker.cap.read()
            frame = cv2.flip(frame, 1)
            self.index_pos = None
            self.index_pos, hand_detected, thumb_up, thumb_down, fingers = hand_tracker.get_hand_position(frame)
            if hand_detected:
                if self.boundaries is not None and self.count > 0:
                    self.index_pos = min(max(self.index_pos[0], self.boundaries[0]), self.boundaries[1]), min(
                        max(self.index_pos[1], self.boundaries[2]), self.boundaries[3])
                # thumb up to start the game in the menu, or confirm the drawing during the game
                if thumb_up and not self.match_done:
                    self.menu = False
                    if self.count > 0:
                        self.check_cell = True

                # open hand to confirm when deciding to return to menu window
                if fingers == [True, True, True, True, True] and self.confirm_window:
                    self.menu = True

                # fist closed to decline return to window
                if fingers == [False, False, False, False, False] and self.confirm_window:
                    self.confirm_window = False

                # when only index finger's up, start drawing
                if fingers == [False, True, False, False, False] and not self.menu and not self.match_done and not self.confirm_window:
                    if not self.isOccupied(self.grid_array, self.index_pos):
                        if not self.drawStart:
                            self.startCell = self.get_cell(self.index_pos)
                            self.drawStart = True
                            self.drawNumber += 1
                            self.draws.append([])
                            self.count += 1
                        self.boundaries = self.get_boundaries(self.startCell, self.x_coordinates, self.y_coordinates)
                        self.index_pos = min(max(self.index_pos[0], self.boundaries[0]), self.boundaries[1]), min(
                            max(self.index_pos[1], self.boundaries[2]), self.boundaries[3])
                        # print(drawNumber)
                        self.draws[self.drawNumber].append(self.index_pos)
                        self.draw = True
                else:
                    self.drawStart = False

                # gesture to delete last segment
                if fingers == [False, True, True, True, False]:
                    if self.draws:
                        if self.drawNumber >= 0 and self.count > 0:
                            if not self.Erasing:
                                self.draws.pop()
                                self.drawNumber -= 1
                                self.Erasing = True
                                self.count -= 1
                else:
                    self.Erasing = False

                # gesture to start a new game when the game's done
                if (fingers == [False, True, False, False, True] and self.match_done) or (
                        fingers == [True, True, True, True, True] and self.confirm_window):
                    self.new_game()

                # Se vuoi uscire nel menù
                if thumb_down and not self.menu:
                    self.confirm_window = True

            self.update_display()

    def update_display(self):
        if self.menu:
            self.WIN.blit(self.background_image, (0, 0))
            if self.index_pos:
                pygame.draw.circle(self.WIN, (255, 0, 0), self.index_pos, 6)
            draw_menu(self.WIN, True, self.WIDTH, self.HEIGHT)
        else:
            if self.check_cell:
                self.o_prob, self.x_prob = self.prob_X_O()
                self.first_move(x_prob, o_prob)
                if o_prob > self.P_MIN and (self.turn % 2) == 0:
                    ris = self.calculate_point(o_prob)
                    self.puntO = (self.puntO + ris) / 2 if self.puntO != 0 else ris
                    self.puntO = round(self.puntO, 2)
                    self.insert_move(self.grid_array, self.startCell, self.chars)
                    self.winner, self.winning_cells = check_winner(self.grid_array)
                    x_prob = o_prob = None
                elif x_prob > self.P_MIN and (self.turn % 2) == 1:
                    ris = self.calculate_point(x_prob)
                    self.puntX = (self.puntX + ris) / 2 if self.puntX != 0 else ris
                    self.puntX = round(self.puntX, 2)
                    self.insert_move(self.grid_array, self.startCell, self.chars)
                    self.winner, self.winning_cells = check_winner(self.grid_array)
                    self.x_prob = self.o_prob = None
                else:
                    self.remove_draw(self.count)
                    self.drawNumber -= self.count
                self.count = 0
                self.check_cell = False

            self.WIN.blit(self.grid_img, (0, 0))
            draw_game(self.WIN, self.index_pos, self.draw, self.draws, self.count, self.turn, self.x_prob, self.o_prob, self.P_MIN, self.puntX, self.puntO, self.winning_cells, self.winner, self.match_done)
            if self.confirm_window:
                draw_confirm_window(self.WIN, self.WIDTH, self.HEIGHT, self.background_image_small)
            if self.winner:
                self.match_done = True
            pygame.display.flip()
    
    def new_game(self):
        self.match_done = False
        self.draws = [[]]
        self.drawNumber = 0
        self.grid_array = [["" for _ in range(3)] for _ in range(3)]
        self.puntO = 0
        self.puntX = 0
        self.count = 0
        self.o_prob = None
        self.x_prob = None
        self.confirm_window = False
        self.winner = None
        self.winning_cells = None
        self.turn = 0
    
    def prob_X_O(self):
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        # x, y, width, height = 192, 53, 199, 167
        boundaries = self.get_boundaries(self.startCell, self.x_coordinates, self.y_coordinates)
        x, y = boundaries[0], boundaries[2]
        width, height = boundaries[1] - boundaries[0], boundaries[3] - boundaries[2]

        subsurface = self.WIN.subsurface((x, y, width, height))
        subsurface_array = pygame.surfarray.array3d(subsurface)
        subsurface_array = np.transpose(subsurface_array, (1, 0, 2))
        cropped_image = Image.fromarray(subsurface_array)

        print(self.startCell)
        model = OurCNN().to(device)
        model.load_state_dict(torch.load(self.MODELPATH, map_location=torch.device('cpu')))
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

    def first_move(self, x_prob, o_prob):
        if self.turn == 0 and x_prob > o_prob:
            self.turn = 1

    def calculate_point(self, prob):
        return 1 + (((prob - self.P_MIN) * 99) / (1 - self.P_MIN))

    def insert_move(self, grid, cell_index, chars):
        i = cell_index // 3
        j = cell_index % 3
        if grid[i][j] == "":
            grid[i][j] = chars[(self.turn % 2)]
            self.turn += 1

        print(grid)

    def remove_draw(self):
        while self.count > 0:
            self.draws.pop()
            self.count -= 1
    
    def get_boundaries(index, xs, ys):
        i = index // 3
        j = index % 3
        return [xs[j], xs[j + 1], ys[i], ys[i + 1]]

    # get the cell index
    def get_cell(xy):
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
        
    def isOccupied(self, grid, index_pos):
        startCell = self.get_cell(index_pos)
        i = startCell // 3
        j = startCell % 3
        if grid[i][j] == "":
            return False
        else:
            return True    
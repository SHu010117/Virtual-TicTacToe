from hand_tracker import HandTracker
from tic_tac_toe import TicTacToeGame

if __name__ == "__main__":
    game = TicTacToeGame()
    hand_tracker = HandTracker()
    game.run(hand_tracker)
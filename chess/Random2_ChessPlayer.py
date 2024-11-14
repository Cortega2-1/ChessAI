'''
CPSC 415 -- Homework #3 support file
Stephen Davies, University of Mary Washington, fall 2017
'''

import random

from chess_player import ChessPlayer

class Random2_ChessPlayer(ChessPlayer):

    def __init__(self, board, color):
        super().__init__(board, color)

    def get_move(self, your_remaining_time, opp_remaining_time, prog_stuff):
        for move in self.board.get_all_available_legal_moves(self.color):
            values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}
            if (self.board.get(move[1])):
                piece = self.board.get(move[1]).get_notation()
                print(piece)
                print(str(piece).upper())
                print(values.get(str(piece).upper(), 0))
            if(self.board.get(move[1])):
                return move
        return random.choice(self.board.get_all_available_legal_moves(self.color))


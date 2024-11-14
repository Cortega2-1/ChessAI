'''
CPSC 415 -- Homework #3 support file
Stephen Davies, University of Mary Washington, fall 2017
'''

import random
from copy import deepcopy

from chess_player import ChessPlayer

class Random3_ChessPlayer(ChessPlayer):

    def __init__(self, board, color):
        super().__init__(board, color)

    def get_move(self, your_remaining_time, opp_remaining_time, prog_stuff):
        opponent_color = 'black'
        if (self.color == 'black'):
            opponent_color = 'white'

        print(opponent_color)
        for move in self.board.get_all_available_legal_moves(self.color):
            boardCopy = deepcopy(self.board)
            boardCopy.make_move(*move)
            if (boardCopy.is_king_in_checkmate(opponent_color)):
                return move
            if (boardCopy.is_king_in_check(opponent_color)):
                return move

        for move in self.board.get_all_available_legal_moves(self.color):
            print(self.board.get(move[1]))
            if(self.board.get(move[1])):
                return move
        return random.choice(self.board.get_all_available_legal_moves(self.color))


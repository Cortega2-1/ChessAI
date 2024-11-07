from chess_player import ChessPlayer
from random import choice
from copy import deepcopy
import math

class CastleVania_ChessPlayer(ChessPlayer):
    PIECE_VALUES = {
        'K': 0, 'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1,
        'k': 0, 'q': 9, 'r': 5, 'b': 3, 'n': 3, 'p': 1
    }

    def __init__(self, board, color):
        super().__init__(board, color)

    def get_move(self, your_remaining_time, opp_remaining_time, prog_stuff):
        # Set a maximum depth for the Minimax based on available time
        depth = 2 if your_remaining_time > 30 else 1
        
        best_move = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        # Use Minimax to find the best move within the depth limit
        for move in self.board.get_all_available_legal_moves(self.color):
            new_board = deepcopy(self.board)
            new_board.make_move(*move)
            move_value = self.minimax(new_board, depth - 1, alpha, beta, False)
            if move_value > best_value:
                best_value = move_value
                best_move = move
            alpha = max(alpha, best_value)
        
        return best_move

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        if depth == 0 or board.is_king_in_checkmate(self.color):
            return self.evaluate_board(board)

        if is_maximizing:
            max_eval = -math.inf
            for move in board.get_all_available_legal_moves(self.color):
                new_board = deepcopy(board)
                new_board.make_move(*move)
                eval = self.minimax(new_board, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            opponent_color = 'black' if self.color == 'white' else 'white'
            for move in board.get_all_available_legal_moves(opponent_color):
                new_board = deepcopy(board)
                new_board.make_move(*move)
                eval = self.minimax(new_board, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_board(self, board):
        score = 0
        for location, piece in board.items():
            piece_value = self.PIECE_VALUES.get(piece.get_notation(), 0)
            if piece.color == self.color:
                score += piece_value
            else:
                score -= piece_value

        # Add a small penalty if the current player's king is in check
        if board.is_king_in_check(self.color):
            score -= 5

        return score

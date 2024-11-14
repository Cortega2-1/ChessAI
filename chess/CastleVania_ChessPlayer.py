from chess_player import ChessPlayer
from random import choice
from copy import deepcopy
import math

class CastleVania_ChessPlayer(ChessPlayer):
    #a dictionary defining the relative value of each chess piece.
    #capital letters represent white pieces, and lowercase letters represent black pieces.
    PIECE_VALUES = {
        'K': 0, 'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1,  #white pieces
        'k': 0, 'q': 9, 'r': 5, 'b': 3, 'n': 3, 'p': 1   #black pieces
    }

    def __init__(self, board, color):
        """
        Initializes the ChessPlayer with the given board state and player's color.
        :param board: The current state of the chess board.
        :param color: The color of the player ('white' or 'black').
        """
        super().__init__(board, color)

    def get_move(self, your_remaining_time, opp_remaining_time, prog_stuff):
        """
        Determines the best move using the Minimax algorithm with Alpha-Beta Pruning.
        The depth of the search is adjusted based on the time remaining.

        :param your_remaining_time: Remaining time for this player.
        :param opp_remaining_time: Remaining time for the opponent.
        :param prog_stuff: Any additional program-specific data (not used here).
        :return: The best move determined by the algorithm.
        """
        #set the maximum search depth based on remaining time.
        #if we have more than 30 seconds, search deeper (depth = 2).
        #otherwise, keep it shallow (depth = 1) to avoid timeouts.
        depth = 2 if your_remaining_time > 30 else 1
        
        #variables to track the best move and its evaluation score.
        best_move = None
        best_value = -math.inf #start with the lowest possible value.
        alpha = -math.inf #alpha value for pruning (maximizing player).
        beta = math.inf #beta value for pruning (minimizing player).

        #loop through all legal moves for the current player.
        for move in self.board.get_all_available_legal_moves(self.color):
            #create a deep copy of the board to simulate the move.
            new_board = deepcopy(self.board)
            new_board.make_move(*move) #make the move on the copied board.

            #use the Minimax algorithm to evaluate this move.
            move_value = self.minimax(new_board, depth - 1, alpha, beta, False)
            
            #if the move is better than our current best move, update it.
            if move_value > best_value:
                best_value = move_value
                best_move = move
            
            #update alpha (best score for maximizing player so far).
            alpha = max(alpha, best_value)
        
        #return the best move found.
        return best_move

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        """
        The Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

        :param board: The current state of the chess board.
        :param depth: The remaining depth to search.
        :param alpha: The best value the maximizing player can guarantee.
        :param beta: The best value the minimizing player can guarantee.
        :param is_maximizing: Boolean indicating if the current player is maximizing.
        :return: The evaluation score of the board.
        """
        #base case: if we've reached the maximum depth or a terminal state (checkmate).
        if depth == 0 or board.is_king_in_checkmate(self.color):
            return self.evaluate_board(board)

        #maximizing player's turn (this player's color).
        if is_maximizing:
            max_eval = -math.inf #start with the lowest possible value.

            #iterate through all legal moves for the current player.
            for move in board.get_all_available_legal_moves(self.color):
                new_board = deepcopy(board)
                new_board.make_move(*move)
                
                #recursively call minimax for the opponent's turn (minimizing).
                eval = self.minimax(new_board, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                
                #update alpha and check if we can prune the search tree.
                alpha = max(alpha, eval)
                if beta <= alpha: #beta cut-off
                    break
            
            return max_eval
        
        #minimizing player's turn (opponent's color).
        else:
            min_eval = math.inf #start with the highest possible value.
            opponent_color = 'black' if self.color == 'white' else 'white'

            #iterate through all legal moves for the opponent.
            for move in board.get_all_available_legal_moves(opponent_color):
                new_board = deepcopy(board)
                new_board.make_move(*move)
                
                #recursively call minimax for our turn (maximizing).
                eval = self.minimax(new_board, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                
                #update beta and check if we can prune the search tree.
                beta = min(beta, eval)
                if beta <= alpha: #alpha cut-off
                    break
            
            return min_eval

    def evaluate_board(self, board):
        """
        Evaluates the board state from the perspective of the current player.
        Positive scores are good for the player, and negative scores are good for the opponent.

        :param board: The current state of the chess board.
        :return: A numerical evaluation of the board.
        """
        score = 0 #initialize the score for the board evaluation.

        #iterate through all pieces on the board and calculate the score.
        for location, piece in board.items():
            #get the value of the piece based on its type.
            piece_value = self.PIECE_VALUES.get(piece.get_notation(), 0)
            
            #if the piece belongs to this player, add its value to the score.
            #if it belongs to the opponent, subtract its value from the score.
            if piece.color == self.color:
                score += piece_value
            else:
                score -= piece_value

        #add a penalty if the current player's king is in check, to encourage moves
        #that protect the king.
        if board.is_king_in_check(self.color):
            score -= 5

        return score

from chess_player import ChessPlayer
from random import choice
from copy import deepcopy
import math

class CastleVania_ChessPlayer(ChessPlayer):
    #a dictionary defining the relative value of each chess piece.
    #the rationale behind these values is based on standard chess theory, where
    #queens are most powerful (9 points), rooks (5), knights/bishops (3), and pawns (1).
    #kings are given a value of 0 because their capture ends the game, making their "value"
    #beyond mere point-scoring.
    PIECE_VALUES = {
        'K': 0, 'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1,
        'k': 0, 'q': 9, 'r': 5, 'b': 3, 'n': 3, 'p': 1
    }

    def __init__(self, board, color):
        """
        initializes the ChessPlayer with the given board state and player's color.
        """
        super().__init__(board, color)

    def get_move(self, your_remaining_time, opp_remaining_time, prog_stuff):
        """
        chooses the best move using the Minimax algorithm with Alpha-Beta Pruning.
        the depth of the search is dynamically adjusted based on remaining time to balance
        decision quality with response speed, especially crucial in time-limited games.
        """
        #if we have plenty of time (more than 30 seconds), we can afford to search deeper (depth=2),
        #which may lead to better strategic decisions. Otherwise, we limit depth to 1 to avoid timing out.

        depth = 2 if your_remaining_time > 30 else 1
        
        best_move = None
        best_value = -math.inf  #initialize with the lowest possible value.
        alpha = -math.inf       #alpha: best score achievable by maximizing player.
        beta = math.inf         #beta: best score achievable by minimizing player.

        #loop through all legal moves for the current player to evaluate the best option.
        for move in self.board.get_all_available_legal_moves(self.color):
            #deep copy of the board to simulate this move without altering the original board state.
            new_board = deepcopy(self.board)
            new_board.make_move(*move)

            #use Minimax to evaluate the board after making this move.
            #we pass `False` because after our move, it's the opponent's turn (minimizing).
            move_value = self.minimax(new_board, depth - 1, alpha, beta, False)
            
            #update best move if this move results in a higher evaluation score.
            if move_value > best_value:
                best_value = move_value
                best_move = move

            #update alpha to reflect the best score found so far for the maximizing player.
            #this helps prune branches where the opponent would avoid these moves.
            alpha = max(alpha, best_value)
        
        return best_move

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        """
        implements the Minimax algorithm with Alpha-Beta Pruning.
        we use this to decide the best possible move for the current player, while also assuming
        the opponent will play optimally to minimize our advantage.
        """
        #if we've reached the maximum search depth or a terminal state (like checkmate),
        #we evaluate the board to decide how favorable it is.
        if depth == 0 or board.is_king_in_checkmate(self.color):
            return self.evaluate_board(board)

        if is_maximizing:
            #maximizing player's turn: we want the highest possible score.
            max_eval = -math.inf

            for move in board.get_all_available_legal_moves(self.color):
                new_board = deepcopy(board)
                new_board.make_move(*move)
                
                #recursively call minimax for the opponent's turn (minimizing).
                eval = self.minimax(new_board, depth - 1, alpha, beta, False)
                
                #we're trying to maximize the evaluation score.
                max_eval = max(max_eval, eval)
                
                #update alpha and check if we can prune (opponent won't let us achieve a higher score).
                alpha = max(alpha, eval)
                if beta <= alpha: #beta cut-off
                    break
            
            return max_eval
        else:
            #minimizing player's turn: we want the lowest possible score.
            min_eval = math.inf
            opponent_color = 'black' if self.color == 'white' else 'white'

            for move in board.get_all_available_legal_moves(opponent_color):
                new_board = deepcopy(board)
                new_board.make_move(*move)
                
                #recursively call minimax for our turn (maximizing).
                eval = self.minimax(new_board, depth - 1, alpha, beta, True)
                
                #we're trying to minimize the evaluation score for the opponent.
                min_eval = min(min_eval, eval)
                
                #update beta and check if we can prune (we won't let them achieve a lower score).
                beta = min(beta, eval)
                if beta <= alpha: #alpha cut-off
                    break
            
            return min_eval

    def evaluate_board(self, board):
        """
        evaluates the current board state. Positive scores are good for our player,
        and negative scores are good for the opponent.

        the evaluation is based on the sum of the piece values on the board, considering:
        - material balance: Higher value pieces contribute more to the score.
        - king safety: A penalty is applied if the king is in check, encouraging defensive moves.
        """
        score = 0

        opponent_color = 'black'
        if (self.color == 'black'):
            opponent_color = 'white'

        #loop through all pieces on the board to calculate the total score.
        for location, piece in board.items():
            #get the value of the piece using our predefined scoring system.
            piece_value = self.PIECE_VALUES.get(piece.get_notation(), 0)
            
            #add the piece value if it's ours, otherwise subtract if it's the opponent's.
            if piece.color == self.color:
                score += piece_value
            else:
                score -= piece_value

        #apply a penalty if our king is in check, to prioritize moves that protect the king.
        if board.is_king_in_check(self.color):
            score -= 5
        # Check if the opponent's king is in check or checkmate.
        if board.is_king_in_check(opponent_color):
            score += 10  # A bonus for putting the opponent's king in check.

        if board.is_king_in_checkmate(opponent_color):
            score += 50  # A large bonus for checkmate (winning).
        
        return score

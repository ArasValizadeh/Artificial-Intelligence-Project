from os import name as os_name
from os import system as os_system
from os.path import abspath, join, dirname
import copy
from main import make_move, set_banners,calculate_winner

def find_varys(cards):
    '''
    This function finds the location of Varys on the board.

    Parameters:
        cards (list): list of Card objects

    Returns:
        varys_location (int): location of Varys
    '''

    varys = [card for card in cards if card.get_name() == 'Varys']

    varys_location = varys[0].get_location()

    return varys_location


def get_valid_moves(cards):
    '''
    This function gets the possible moves for the player.

    Parameters:
        cards (list): list of Card objects

    Returns:
        moves (list): list of possible moves
    '''

    # Get the location of Varys
    varys_location = find_varys(cards)

    # Get the row and column of Varys
    varys_row, varys_col = varys_location // 6, varys_location % 6

    moves = []

    # Get the cards in the same row or column as Varys
    for card in cards:
        if card.get_name() == 'Varys':
            continue

        row, col = card.get_location() // 6, card.get_location() % 6

        if row == varys_row or col == varys_col:
            moves.append(card.get_location())

    return moves


def evaluate_state(cards, player1, player2):
    """
    heuristic evaluation function to give score for each state in game.
    """
    banner_weights = {
        'Stark': 8,'Greyjoy': 6,'Lannister': 6,'Targaryen': 6,'Baratheon': 5,'Tyrell': 4,'Tully': 4
    }
    banner_cards = {
        'Stark': 8,'Greyjoy': 7,'Lannister': 6,'Targaryen': 5,'Baratheon': 4,'Tyrell': 3,'Tully': 2
    }
    
    score = 0

    #calculate score for player 1
    for house in player1.get_banners():
        if player1.get_banners()[house]:
            score += banner_weights[house] * 4
            num_cards = 0
            for card in player1.get_cards():
                if card == house:
                    num_cards += 1
            if num_cards <= banner_cards[house] // 2 + 1:
                score += num_cards * 2

    # Calculate score for Player 2
    for house in player2.get_banners():
        if player2.get_banners()[house]:
            score -= banner_weights[house] * 4
            num_cards = 0
            for card in player1.get_cards():
                if card == house:
                    num_cards += 1
            if num_cards > banner_cards[house] // 2 + 1:
                penalty = -(banner_cards[house] // 2)
            else:
                penalty = num_cards
            score -= penalty * 2
    return score

def minimax(cards, player1, player2, depth, maximizing_player, alpha, beta, deep_search=False):
    """
    Minimax implementation with optional deep_search mode and Alpha-Beta pruning.
    """
    if depth == 0 or len(get_valid_moves(cards)) == 0:
        if deep_search:
            if calculate_winner(player1, player2) == 1:
                return 1000, None
            else:
                return -1000, None
        return evaluate_state(cards, player1, player2), None

    best_move = None
    if maximizing_player:
        best_eval = float('-inf')
        for move in get_valid_moves(cards):
            new_cards, new_player1, new_player2 = simulate_move(cards, player1, player2, move, player=1)
            eval_score, _ = minimax(new_cards, new_player1, new_player2, depth - 1, False, alpha, beta, deep_search)
            if eval_score > best_eval:
                best_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Prune the branch
    else:
        best_eval = float('inf')
        for move in get_valid_moves(cards):
            new_cards, new_player1, new_player2 = simulate_move(cards, player1, player2, move, player=2)
            eval_score, _ = minimax(new_cards, new_player1, new_player2, depth - 1, True, alpha, beta, deep_search)
            if eval_score < best_eval:
                best_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Prune the branch

    return best_eval, best_move


def simulate_move(cards, player1, player2, move, player):
    """
    Simulate a move and return the new state of the game.
    """
    new_cards = copy.deepcopy(cards)
    new_player1 = copy.deepcopy(player1)
    new_player2 = copy.deepcopy(player2)

    selected_house = make_move(new_cards, move, new_player1 if player == 1 else new_player2)
    set_banners(new_player1, new_player2, selected_house, 1 if player == 1 else 2)

    return new_cards, new_player1, new_player2

def get_move(cards, player1, player2):
    """
    Get the best move using the Minimax function with Alpha-Beta pruning.
    """
    depth = 7 if len(cards) < 20 else 5

    _, best_move = minimax(
        cards, player1, player2, 
        depth=depth, maximizing_player=True, 
        alpha=float('-inf'), beta=float('inf'), 
        deep_search=len(cards) <= 16
    )

    return best_move
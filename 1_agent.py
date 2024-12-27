import math
import copy
from classes import Card, Player
from main import set_banners  # Importing set_banners function

def find_varys(cards):
    '''
    Finds the location of Varys on the board.

    Parameters:
        cards (list): list of Card objects

    Returns:
        varys_location (int): location of Varys
    '''
    varys = [card for card in cards if card.get_name() == 'Varys']
    return varys[0].get_location()

def get_valid_moves(cards):
    '''
    Gets the possible moves for the player.

    Parameters:
        cards (list): list of Card objects

    Returns:
        moves (list): list of possible moves
    '''
    varys_location = find_varys(cards)
    varys_row, varys_col = varys_location // 6, varys_location % 6
    moves = []

    for card in cards:
        if card.get_name() == 'Varys':
            continue

        row, col = card.get_location() // 6, card.get_location() % 6
        if row == varys_row or col == varys_col:
            moves.append(card.get_location())

    return moves


def evaluate_state(cards, player1, player2):
    """
    Evaluates the game state and returns a score. Higher score favors player1.
    """
    score = 0
    banner_weights = {
        'Stark': 10, 'Greyjoy': 8, 'Lannister': 6,
        'Targaryen': 5, 'Baratheon': 4, 'Tyrell': 3, 'Tully': 2
    }

    # Evaluate banners and card counts
    for house in banner_weights.keys():
        if player1.get_banners().get(house):
            score += banner_weights[house] * 4
            score += len(player1.get_cards()[house]) * 2
        if player2.get_banners().get(house):
            score -= banner_weights[house] * 4
            score -= len(player2.get_cards()[house]) * 2

    return score

def simulate_move(cards, move, player1, player2, is_maximizing):
    """
    Simulates a move and returns the updated game state.
    """
    simulated_cards = copy.deepcopy(cards)
    current_player = copy.deepcopy(player1 if is_maximizing else player2)
    opponent_player = copy.deepcopy(player2 if is_maximizing else player1)

    # Capture the card and update the state
    card_to_capture = next((card for card in simulated_cards if card.get_location() == move), None)
    if card_to_capture:
        card_to_capture.set_location(-1)
        current_player.add_card(card_to_capture)
        selected_house = card_to_capture.get_house()
        set_banners(current_player, opponent_player, selected_house, 1 if is_maximizing else 2)

    return simulated_cards, current_player, opponent_player

def minimax(cards, player1, player2, depth, is_maximizing, alpha, beta):
    """
    Implements the minimax algorithm with Alpha-Beta pruning.
    """
    if depth == 0 or not get_valid_moves(cards):
        return evaluate_state(cards, player1, player2), None

    best_score = -math.inf if is_maximizing else math.inf
    best_move = None

    for move in get_valid_moves(cards):
        simulated_cards, new_player1, new_player2 = simulate_move(cards, move, player1, player2, is_maximizing)
        score, _ = minimax(simulated_cards, new_player1, new_player2, depth - 1, not is_maximizing, alpha, beta)

        if is_maximizing:
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
        else:
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, score)

        if beta <= alpha:
            break  # Prune the branch

    return best_score, best_move

def get_move(cards, player1, player2):
    """
    Determines the best move using the Minimax algorithm.
    """
    depth = 7 if len(cards) < 20 else 5
    _, best_move = minimax(cards, player1, player2, depth, is_maximizing=True, alpha=-math.inf, beta=math.inf)
    return best_move
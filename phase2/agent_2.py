import copy
import random
from itertools import combinations
from math import inf
from main import make_move, make_companion_move, set_banners, calculate_winner, house_card_count, remove_unusable_companion_cards
from utils.classes import Card, Player

def find_varys(cards):
    """Finds the location of Varys on the board."""
    varys = [card for card in cards if card.get_name() == "Varys"]
    return varys[0].get_location()

def get_valid_moves(cards):
    """Gets all possible moves for Varys."""
    varys_location = find_varys(cards)
    varys_row, varys_col = varys_location // 6, varys_location % 6
    return [card.get_location() for card in cards if card.get_name() != "Varys" and (card.get_location() // 6 == varys_row or card.get_location() % 6 == varys_col)]

def get_valid_ramsay(cards):
    """Returns valid move locations for Ramsay's effect."""
    return [card.get_location() for card in cards]

def get_valid_jon_sandor_jaqan(cards):
    """Returns valid move locations for Jon Snow, Sandor Clegane, and Jaqen."""
    return [card.get_location() for card in cards if card.get_name() != "Varys"]

def evaluate_state(cards, player1, player2):
    """Evaluates the board state for Minimax."""
    banner_weights = {'Stark': 9, 'Greyjoy': 7, 'Lannister': 7, 'Targaryen': 6, 'Baratheon': 5, 'Tyrell': 9, 'Tully': 10}
    score = 0

    # Evaluate banners for both players
    for house in player1.get_banners():
        if player1.get_banners()[house]:
            score += banner_weights[house] * 10
            score += len(player1.get_cards()[house]) * 2

    for house in player2.get_banners():
        if player2.get_banners()[house]:
            score -= banner_weights[house] * 10
            score -= len(player2.get_cards()[house]) * 2

    # Favor having more moves available
    score += len(get_valid_moves(cards)) * 3
    return score

def simulate_move(cards, player1, player2, move, player):
    """Simulates a normal move."""
    if isinstance(move, list):  # Companion move
        return simulate_companion_move(cards, player1, player2, move, player)
    
    new_cards = copy.deepcopy(cards)
    current_player = player1 if player == 1 else player2
    selected_house = make_move(new_cards, move, current_player)
    set_banners(player1, player2, selected_house, player)
    return new_cards, player1, player2

def simulate_companion_move(cards, player1, player2, move, player):
    """Simulates companion move effects."""
    selected_companion = move[0]
    new_cards = copy.deepcopy(cards)
    current_player = player1 if player == 1 else player2

    if selected_companion == "Ramsay":
        first_card_loc, second_card_loc = move[1], move[2]
        first_card = next((c for c in new_cards if c.get_location() == first_card_loc), None)
        second_card = next((c for c in new_cards if c.get_location() == second_card_loc), None)
        if first_card and second_card:
            first_card.set_location(second_card_loc)
            second_card.set_location(first_card_loc)

    elif selected_companion == "Sandor":
        selected_card_loc = move[1]
        new_cards = [c for c in new_cards if c.get_location() != selected_card_loc]

    elif selected_companion == "Jaqen":
        first_card_loc, second_card_loc, companion_to_remove = move[1], move[2], move[3]
        new_cards = [c for c in new_cards if c.get_location() not in [first_card_loc, second_card_loc]]

    return new_cards, player1, player2

def minimax(cards, player1, player2, companion_cards, choose_companion, depth, is_maximizing, alpha=-inf, beta=inf):
    """Minimax with Alpha-Beta Pruning and Companion Move Support."""
    if depth == 0 or not get_valid_moves(cards):
        return evaluate_state(cards, player1, player2), None

    optimal_move = None

    # Maximizing Player (Our AI)
    if is_maximizing:
        max_eval = -inf
        for move in get_valid_moves(cards):
            next_cards, next_player1, next_player2 = simulate_move(cards, player1, player2, move, 1)
            current_eval, _ = minimax(next_cards, next_player1, next_player2, companion_cards, choose_companion, depth - 1, False, alpha, beta)
            if current_eval > max_eval:
                max_eval, optimal_move = current_eval, move
            alpha = max(alpha, current_eval)
            if beta <= alpha:
                break
        return max_eval, optimal_move

    # Minimizing Player (Opponent)
    else:
        min_eval = inf
        for move in get_valid_moves(cards):
            next_cards, next_player1, next_player2 = simulate_move(cards, player1, player2, move, 2)
            current_eval, _ = minimax(next_cards, next_player1, next_player2, companion_cards, choose_companion, depth - 1, True, alpha, beta)
            if current_eval < min_eval:
                min_eval, optimal_move = current_eval, move
            beta = min(beta, current_eval)
            if beta <= alpha:
                break
        return min_eval, optimal_move

def select_best_companion_move(cards, player1, player2, companion_cards):
    """Selects the best companion move using Minimax."""
    best_companion_move = None
    max_eval = -inf

    for selected_companion in companion_cards.keys():
        move = [selected_companion]
        choices = companion_cards[selected_companion]['Choice']
        valid_moves = get_valid_jon_sandor_jaqan(cards)

        if choices == 1 and valid_moves:
            move.append(random.choice(valid_moves))
        elif choices == 2 and len(valid_moves) >= 2:
            move.extend(random.sample(valid_moves, 2))
        elif choices == 3 and len(valid_moves) >= 2:
            move.extend(random.sample(valid_moves, 2))
            move.append(random.choice(list(companion_cards.keys())))

        eval_score, _ = minimax(cards, player1, player2, companion_cards, False, 3, True)
        if eval_score > max_eval:
            max_eval = eval_score
            best_companion_move = move

    return best_companion_move

def get_move(cards, player1, player2, companion_cards, choose_companion):
    """Determines the best move, either normal or companion-based."""
    if choose_companion and companion_cards:
        return select_best_companion_move(cards, player1, player2, companion_cards)

    return minimax(cards, player1, player2, companion_cards, choose_companion, 5, True)[1]
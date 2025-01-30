from os.path import abspath, join, dirname
import copy
import random
from math import inf
from main import make_move, set_banners, calculate_winner
from utils.classes import Card, Player

# Finds the location of Varys on the board
def find_varys(cards):
    varys = [card for card in cards if card.get_name() == 'Varys']
    return varys[0].get_location()

# Returns possible moves for Varys
def get_valid_moves(cards):
    varys_location = find_varys(cards)
    varys_row, varys_col = varys_location // 6, varys_location % 6
    return [card.get_location() for card in cards if card.get_name() != 'Varys' and (card.get_location() // 6 == varys_row or card.get_location() % 6 == varys_col)]

# Returns valid moves for companion effects
def get_valid_ramsay(cards):
    return [card.get_location() for card in cards]

def get_valid_jon_sandor_jaqan(cards):
    return [card.get_location() for card in cards if card.get_name() != 'Varys']

# Evaluates the board state
def evaluate_state(cards, player1, player2):
    banner_weights = {'Stark': 9, 'Greyjoy': 7, 'Lannister': 7, 'Targaryen': 6, 'Baratheon': 5, 'Tyrell': 9, 'Tully': 10}
    score = 0

    for house in player1.get_banners():
        if player1.get_banners()[house]:
            score += banner_weights[house] * 10
            num_cards = sum(1 for card in player1.get_cards() if card == house)
            score += num_cards * 2 if num_cards <= 4 else num_cards

    for house in player2.get_banners():
        if player2.get_banners()[house]:
            score -= banner_weights[house] * 10
            num_cards = sum(1 for card in player2.get_cards() if card == house)
            score -= num_cards * 2 if num_cards <= 4 else num_cards

    return score + (len(get_valid_moves(cards)) * 3)

# Simulates companion move effects
def simulate_companion_move(cards, player1, player2, move, player):
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

# Simulates a move for Minimax
def simulate_move(cards, player1, player2, move, player):
    if isinstance(move, list):
        return simulate_companion_move(cards, player1, player2, move, player)
    
    new_cards = copy.deepcopy(cards)
    current_player = player1 if player == 1 else player2
    selected_house = make_move(new_cards, move, current_player)
    set_banners(player1, player2, selected_house, player)
    return new_cards, player1, player2

# Minimax with Alpha-Beta Pruning (Now Includes Companion Moves)
def minimax(cards, player1, player2, companion_cards, depth, is_maximizing, alpha=-inf, beta=inf):
    if depth == 0 or not get_valid_moves(cards):
        return evaluate_state(cards, player1, player2), None

    optimal_move = None

    # Maximizing Player (Our AI)
    if is_maximizing:
        max_eval = -inf
        for move in get_valid_moves(cards):
            next_cards, next_player1, next_player2 = simulate_move(cards, player1, player2, move, 1)
            current_eval, _ = minimax(next_cards, next_player1, next_player2, companion_cards, depth - 1, False, alpha, beta)
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
            current_eval, _ = minimax(next_cards, next_player1, next_player2, companion_cards, depth - 1, True, alpha, beta)
            if current_eval < min_eval:
                min_eval, optimal_move = current_eval, move
            beta = min(beta, current_eval)
            if beta <= alpha:
                break
        return min_eval, optimal_move

# Selects the best move (Normal or Companion)
def get_move(cards, player1, player2, companion_cards, choose_companion):
    if choose_companion and companion_cards:
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

            eval_score, _ = minimax(cards, player1, player2, companion_cards, 3, True)
            if eval_score > max_eval:
                max_eval = eval_score
                best_companion_move = move

        return best_companion_move

    return minimax(cards, player1, player2, companion_cards, 5, True)[1]
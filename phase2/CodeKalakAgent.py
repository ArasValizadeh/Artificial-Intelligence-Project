from os.path import abspath, join, dirname
import copy
import random
from math import inf
from main import make_move, set_banners, calculate_winner
from utils.classes import Card, Player

def find_varys(cards):
    varys = [card for card in cards if card.get_name() == 'Varys']
    return varys[0].get_location()

def get_valid_moves(cards):
    varys_location = find_varys(cards)
    varys_row, varys_col = varys_location // 6, varys_location % 6
    moves = [card.get_location() for card in cards if card.get_name() != 'Varys' and (card.get_location() // 6 == varys_row or card.get_location() % 6 == varys_col)]
    return moves

def get_valid_ramsay(cards):
    return [card.get_location() for card in cards]

def get_valid_jon_sandor_jaqan(cards):
    return [card.get_location() for card in cards if card.get_name() != 'Varys']

def evaluate_state(cards, player1, player2):
    banner_weights = {'Stark': 9, 'Greyjoy': 7, 'Lannister': 7, 'Targaryen': 6, 'Baratheon': 5, 'Tyrell': 9, 'Tully': 10}
    banner_cards = {'Stark': 8, 'Greyjoy': 7, 'Lannister': 6, 'Targaryen': 5, 'Baratheon': 4, 'Tyrell': 3, 'Tully': 2}
    score = 0

    for house in player1.get_banners():
        if player1.get_banners()[house]:
            score += banner_weights[house] * 10
            num_cards = sum(1 for card in player1.get_cards() if card == house)
            if num_cards <= banner_cards[house] // 2 + 1:
                score += num_cards * 2

    for house in player2.get_banners():
        if player2.get_banners()[house]:
            score -= banner_weights[house] * 10
            num_cards = sum(1 for card in player2.get_cards() if card == house)
            penalty = -(banner_cards[house] // 2) if num_cards > banner_cards[house] // 2 + 1 else num_cards
            score -= penalty * 2

    player1_moves = len(get_valid_moves(cards))
    player2_moves = len(get_valid_moves(cards))
    score += (player1_moves - player2_moves) * 3
    return score

def minimax(cards, player1, player2, depth, is_maximizing, alpha=-inf, beta=inf, no_heuristic=False):
    valid_moves = get_valid_moves(cards)

    if depth == 0 or not valid_moves:
        if no_heuristic:
            return (2000 if calculate_winner(player1, player2) == 1 else -2000, None)
        return evaluate_state(cards, player1, player2), None

    optimal_move = None
    if is_maximizing:
        max_eval = -inf
        for move in valid_moves:
            next_cards, next_player1, next_player2 = simulate_move(cards, player1, player2, move, 1)
            current_eval, _ = minimax(next_cards, next_player1, next_player2, depth - 1, False, alpha, beta, no_heuristic)
            if current_eval > max_eval:
                max_eval, optimal_move = current_eval, move
            alpha = max(alpha, current_eval)
            if beta <= alpha:
                break
        return max_eval, optimal_move
    else:
        min_eval = inf
        for move in valid_moves:
            next_cards, next_player1, next_player2 = simulate_move(cards, player1, player2, move, 2)
            current_eval, _ = minimax(next_cards, next_player1, next_player2, depth - 1, True, alpha, beta, no_heuristic)
            if current_eval < min_eval:
                min_eval, optimal_move = current_eval, move
            beta = min(beta, current_eval)
            if beta <= alpha:
                break
        return min_eval, optimal_move

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

def simulate_move(cards, player1, player2, move, player):
    if isinstance(move, list):
        return simulate_companion_move(cards, player1, player2, move, player)
    
    new_cards = copy.deepcopy(cards)
    current_player = player1 if player == 1 else player2
    selected_house = make_move(new_cards, move, current_player)
    set_banners(player1, player2, selected_house, player)
    return new_cards, player1, player2

def get_move(cards, player1, player2, companion_cards, choose_companion):
    new_cards = copy.deepcopy(cards)
    new_player1, new_player2 = copy.deepcopy(player1), copy.deepcopy(player2)

    if choose_companion:
        if companion_cards:
            selected_companion = list(companion_cards.keys())[0]
            move = [selected_companion]
            choices = companion_cards[selected_companion]['Choice']

            print(f"\nAgent selecting companion card: {selected_companion}")

            if choices == 1:  # Jon Snow or Sandor
                selected_card = random.choice(get_valid_jon_sandor_jaqan(new_cards))
                move.append(selected_card)
                print(f"🚀 Selected card at position: {selected_card}")
                
            elif choices == 2:  # Ramsay
                valid_moves = get_valid_ramsay(new_cards)
                if len(valid_moves) >= 2:
                    selected_cards = random.sample(valid_moves, 2)
                    move.extend(selected_cards)
                    print(f"🚀 Selected cards to swap at positions: {selected_cards[0]} and {selected_cards[1]}")
                else:
                    move.extend(valid_moves)
                    print(f"🚀 Not enough valid moves for Ramsay, using: {valid_moves}")
                    
            elif choices == 3:  # Jaqen
                valid_moves = get_valid_jon_sandor_jaqan(new_cards)
                if len(valid_moves) >= 2 and len(companion_cards) > 1:
                    selected_cards = random.sample(valid_moves, 2)
                    selected_companion_remove = random.choice([c for c in companion_cards.keys() if c != selected_companion])
                    move.extend(selected_cards)
                    move.append(selected_companion_remove)
                    print(f"🚀 Selected cards to remove at positions: {selected_cards[0]} and {selected_cards[1]}")
                    print(f"🚀 Selected companion card to remove: {selected_companion_remove}")
                else:
                    move.extend(valid_moves)
                    other_companion = random.choice(list(companion_cards.keys())) if companion_cards else None
                    move.append(other_companion)
                    print(f"🚀 Not enough valid moves/companions for Jaqen, using moves: {valid_moves}")
                    print(f"🚀 And companion to remove: {other_companion}")

            print(f"Final companion move: {move}\n")
            return move

    depth = 9 if len(new_cards) < 25 else 5
    flag = len(new_cards) <= 16

    best_move = minimax(
        cards=new_cards,
        player1=new_player1,
        player2=new_player2,
        depth=depth,
        is_maximizing=True,
        alpha=-inf,
        beta=inf,
        no_heuristic=flag
    )[1]

    return best_move
import itertools
import random
import copy
from main import house_card_count, make_companion_move, make_move, remove_unusable_companion_cards, set_banners

def find_varys(cards):
    """Find the location of Varys on the board."""
    varys = [card for card in cards if card.get_name() == 'Varys']
    return varys[0].get_location() if varys else None  # Handle missing Varys safely

def get_valid_moves(cards):
    """Get valid moves where Varys can move."""
    varys_location = find_varys(cards)
    if varys_location is None:
        return []  # Avoid errors if Varys is missing
    
    varys_row, varys_col = varys_location // 6, varys_location % 6
    return [
        card.get_location() for card in cards 
        if card.get_name() != 'Varys' and 
        (card.get_location() // 6 == varys_row or card.get_location() % 6 == varys_col)
    ]

def get_valid_ramsay(cards):
    """Get all possible moves for Ramsay."""
    return [card.get_location() for card in cards] if cards else []

def get_valid_jon_sandor_jaqan(cards):
    """Get possible moves for Jon Snow, Sandor Clegane, and Jaqen H'ghar."""
    return [card.get_location() for card in cards if card.get_name() != 'Varys'] if cards else []

def evaluate_state(cards, player1, player2, companion_cards, turn):
    """Evaluate the game state based on banners, moves, and companions."""
    score = 0
    banner_members = {'Stark': 8, 'Greyjoy': 7, 'Lannister': 6, 'Targaryen': 5, 'Baratheon': 4, 'Tyrell': 3, 'Tully': 2}

    for house, banner in player1.get_banners().items():
        if banner:
            score += 10  # Fixed weight for each banner owned
            num_members = len(player1.get_cards()[house])
            score += num_members * 2 if num_members > banner_members[house] // 2 + 1 else num_members

    for house, banner in player2.get_banners().items():
        if banner:
            score -= 10
            num_members = len(player2.get_cards()[house])
            score -= num_members * 2 if num_members > banner_members[house] // 2 + 1 else num_members

    return score

def minimax(cards, player1, player2, companion_cards, choose_companion, depth, maximizing_player, alpha, beta):
    """Minimax algorithm with Alpha-Beta pruning and optimized companion handling."""
    if depth == 0 or (choose_companion == 0 and not get_valid_moves(cards)) or (choose_companion and not companion_cards):
        return evaluate_state(cards, player1, player2, companion_cards, maximizing_player), None

    best_move = None
    if choose_companion == 0:
        moves = get_valid_moves(cards)
        if not moves:
            return evaluate_state(cards, player1, player2, companion_cards, maximizing_player), None

        if maximizing_player:
            max_eval = float('-inf')
            for move in moves:
                new_cards, new_player1, new_player2, new_companion_cards = (
                    copy.deepcopy(cards), copy.deepcopy(player1), copy.deepcopy(player2), copy.deepcopy(companion_cards)
                )
                selected_house = make_move(new_cards, move, new_player1)
                remove_unusable_companion_cards(new_cards, new_companion_cards)
                set_banners(new_player1, new_player2, selected_house, 1)

                eval_score, _ = minimax(
                    new_cards, new_player1, new_player2, new_companion_cards, 1, depth - 1, True, alpha, beta
                )

                if eval_score > max_eval:
                    max_eval, best_move = eval_score, move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in moves:
                new_cards, new_player1, new_player2, new_companion_cards = (
                    copy.deepcopy(cards), copy.deepcopy(player1), copy.deepcopy(player2), copy.deepcopy(companion_cards)
                )
                selected_house = make_move(new_cards, move, new_player2)
                remove_unusable_companion_cards(new_cards, new_companion_cards)
                set_banners(new_player1, new_player2, selected_house, 2)

                eval_score, _ = minimax(new_cards, new_player1, new_player2, new_companion_cards, True, depth - 1, False, alpha, beta)

                if eval_score < min_eval:
                    min_eval, best_move = eval_score, move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    # Companion move logic
    best_eval = float('-inf') if maximizing_player else float('inf')

    for companion in companion_cards.keys():
        move = [companion]
        choices = companion_cards[companion]['Choice']

        if choices == 1:  # Jon Snow or Sandor
            valid_moves = get_valid_jon_sandor_jaqan(cards)
            if valid_moves:
                move.append(random.choice(valid_moves))
            else:
                continue  # Skip if no valid moves

        elif choices == 2:  # Ramsay
            valid_moves = get_valid_ramsay(cards)
            if len(valid_moves) >= 2:
                move.extend(random.sample(valid_moves, 2))
            else:
                continue  # Skip invalid choices

        elif choices == 3:  # Jaqen
            valid_moves = get_valid_jon_sandor_jaqan(cards)
            if len(valid_moves) >= 2:
                move.extend(random.sample(valid_moves, 2))
            if len(companion_cards) > 1:
                move.append(random.choice(list(companion_cards.keys())))
            else:
                continue  # Skip invalid choices

        new_cards, new_player1, new_player2, new_companion_cards = (
            copy.deepcopy(cards), copy.deepcopy(player1), copy.deepcopy(player2), copy.deepcopy(companion_cards)
        )
        selected_house = make_companion_move(new_cards, new_companion_cards, move, new_player1)
        remove_unusable_companion_cards(new_cards, new_companion_cards)
        set_banners(new_player1, new_player2, selected_house, 1)

        eval_score, _ = minimax(new_cards, new_player1, new_player2, new_companion_cards, 0, depth - 1, False, alpha, beta)

        if (maximizing_player and eval_score > best_eval) or (not maximizing_player and eval_score < best_eval):
            best_eval, best_move = eval_score, move

    return best_eval, best_move

def get_move(cards, player1, player2, companion_cards, choose_companion):
    """Determine the best move for the agent."""
    num_cards = len(cards)
    depth = 3 if num_cards > 20 else 5 if num_cards > 10 else 7  # Dynamic depth

    move = minimax(cards, player1, player2, companion_cards, choose_companion, depth, True, float('-inf'), float('inf'))[1]

    if move is None:
        valid_moves = get_valid_moves(cards)
        move = random.choice(valid_moves) if valid_moves else None  # Fallback

    return move

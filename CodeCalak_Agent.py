import math
from classes import Card, Player
import time
import copy

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

# def evaluate_state(cards, player1, player2):
#     '''
#     Evaluates the game state and returns a score.
#     Higher score favors player1.

#     Parameters:
#         cards (list): list of Card objects
#         player1 (Player): the current player
#         player2 (Player): the opponent

#     Returns:
#         score (int): the evaluation score
#     '''
#     # Define the total cards per house based on the game structure
#     house_total_cards = {
#         "Stark": 8,
#         "Greyjoy": 7,
#         "Lannister": 6,
#         "Targaryen": 5,
#         "Baratheon": 4,
#         "Tyrell": 3,
#         "Tully": 2
#     }

#     score = 0

#     # Add points for completed houses for player1
#     for house, cards_count in player1.get_cards().items():
#         if len(cards_count) == house_total_cards[house]:
#             score += 3 * house_total_cards[house] # Reward for completing a house

#     # Subtract points for completed houses for player2
#     for house, cards_count in player2.get_cards().items():
#         if len(cards_count) == house_total_cards[house]:
#             score -= 3 * house_total_cards[house]  # Penalty for opponent completing a house

#     # Add points for each card captured in favor of player1
#     for house, cards_count in player1.get_cards().items():
#         score += len(cards_count) * 5

#     # Subtract points for each card captured in favor of player2
#     for house, cards_count in player2.get_cards().items():
#         score -= len(cards_count) * 5
    
#     return score


def evaluate_state(cards, player1, player2):
    '''
    Evaluates the game state and returns a score.
    Higher score favors player1.

    Parameters:
        cards (list): list of Card objects
        player1 (Player): the current player
        player2 (Player): the opponent

    Returns:
        score (int): the evaluation score
    '''
    # Define the total cards per house based on the game structure
    house_total_cards = {
        "Stark": 8,
        "Greyjoy": 7,
        "Lannister": 6,
        "Targaryen": 5,
        "Baratheon": 4,
        "Tyrell": 3,
        "Tully": 2
    }

    score = 0    

    # Iterate through each house and determine who owns the flag
    for house, total_cards in house_total_cards.items():
        # if player1.get_banners()[house]:
        #     score += 15
        #     continue
        # if player2.get_banners()[house]:
        #     score -= 15
        #     continue
        
        player1_count = len(player1.get_cards().get(house, []))
        player2_count = len(player2.get_cards().get(house, []))

        # Determine who holds the flag
        if player1_count > player2_count:
            score += total_cards  # Player 1 gets the flag points
        elif player2_count > player1_count:
            score -= total_cards  # Player 2 gets the flag points
            
        

    return score


def simulate_move(cards, move, player1, player2, is_maximizing):
    '''
    Simulates a move and returns the updated game state.

    Parameters:
        cards (list): list of Card objects
        move (int): the move to simulate (location of the card to capture)
        player1 (Player): the current player
        player2 (Player): the opponent
        is_maximizing (bool): whether the current player is maximizing or not

    Returns:
        (list, Player, Player): the updated cards, player1, and player2
    '''
    # Create deep copies of the game state
    simulated_cards = copy.deepcopy(cards)
    new_player1 = copy.deepcopy(player1)
    new_player2 = copy.deepcopy(player2)

    # Determine the current player
    current_player = new_player1 if is_maximizing else new_player2
    opponent_player = new_player2 if is_maximizing else new_player1

    # Find the card at the move location
    card_to_capture = next((card for card in simulated_cards if card.get_location() == move), None)
    if card_to_capture:
        # Update the card's location to a special "captured" area (e.g., -1)
        card_to_capture.set_location(-1)

        # Add the card to the current player's collection
        current_player.add_card(card_to_capture)

        # Check if the current player now controls the banner
        house = card_to_capture.get_house()
        if len(current_player.get_cards()[house]) > len(opponent_player.get_cards()[house]):
            current_player.get_house_banner(house)
            opponent_player.remove_house_banner(house)

    return simulated_cards, new_player1, new_player2


def minimax(cards, player1, player2, depth, is_maximizing, alpha, beta):
    '''
    Implements the minimax algorithm with alpha-beta pruning.

    Parameters:
        cards (list): list of Card objects
        player1 (Player): the current player
        player2 (Player): the opponent
        depth (int): the current depth of the search
        is_maximizing (bool): whether the current player is maximizing or not
        alpha (float): the alpha value for pruning
        beta (float): the beta value for pruning

    Returns:
        best_score (int): the evaluation score of the best move
    '''
    if depth == 0 or not get_valid_moves(cards):
        return evaluate_state(cards, player1, player2)

    if is_maximizing:
        max_eval = -math.inf
        for move in get_valid_moves(cards):
            simulated_cards, new_player1, new_player2 = simulate_move(
                cards, move, player1, player2, is_maximizing
            )
            eval = minimax(simulated_cards, new_player1, new_player2, depth - 1, False, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for move in get_valid_moves(cards):
            simulated_cards, new_player1, new_player2 = simulate_move(
                cards, move, player1, player2, is_maximizing
            )
            eval = minimax(simulated_cards, new_player1, new_player2, depth - 1, True, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def get_move(cards, player1, player2):
    '''
    Determines the best move using both minimax and alpha-beta pruning, and prints timing results.

    Parameters:
        cards (list): list of Card objects
        player1 (Player): the current player
        player2 (Player): the opponent

    Returns:
        move (int): the best move for the player
    '''
    best_move = None

    best_score = -math.inf
    for move in get_valid_moves(cards):
        simulated_cards, new_player1, new_player2 = simulate_move(
            cards, move, player1, player2, True
        )
        move_score = minimax(simulated_cards, new_player1, new_player2, depth=4, is_maximizing=True, alpha=-math.inf, beta=math.inf)
        if move_score > best_score:
            best_score = move_score
            best_move = move

    return best_move
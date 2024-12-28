from os import name as os_name
from os import system as os_system
from os.path import abspath, join, dirname
import copy
from main import make_move, set_banners,calculate_winner
from math import inf

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
    Calculates a heuristic score for the current game state by evaluating the banners
    captured by both players and the number of cards they control.

    Parameters:
        cards (list): A list of `Card` objects representing the current game state.
        player1 (Player): An object representing Player 1, including banners and cards.
        player2 (Player): An object representing Player 2, including banners and cards.

    Returns:
        int: A score indicating the favorability of the game state for Player 1.
            Positive scores favor Player 1; negative scores favor Player 2.
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

def minimax(cards, player1, player2, depth, is_maximizing, alpha=-inf, beta=inf, deep_search=False):
    """
    Uses the Minimax algorithm with Alpha-Beta pruning and optional deep search
    to determine the best move for a given game state.

    Parameters:
        cards (list): Represents the current game board as `Card` objects.
        player1 (Player): Player 1's state including banners and cards.
        player2 (Player): Player 2's state including banners and cards.
        depth (int): Depth limit for game tree exploration.
        is_maximizing (bool): Indicates if the current player is maximizing the score.
        alpha (float): Alpha value for pruning (best score for maximizing player).
        beta (float): Beta value for pruning (best score for minimizing player).
        deep_search (bool, optional): Flag for additional end-game evaluation when no moves remain.

    Returns:
        tuple: Contains:
            - int: The highest or lowest score achievable from this state.
            - int or None: The index of the optimal move, or None if no valid moves are available.
    """
    valid_moves = get_valid_moves(cards)
    if depth == 0 or not valid_moves:
        if deep_search:
            winner_score = 1000 if calculate_winner(player1, player2) == 1 else -1000
            return winner_score, None
        score = evaluate_state(cards, player1, player2)
        return score, None

    optimal_move = None
    if is_maximizing:
        max_eval = -inf
        for move in valid_moves:
            next_cards, next_player1, next_player2 = simulate_move(cards, player1, player2, move, player=1)
            current_eval, _ = minimax(next_cards, next_player1, next_player2, depth - 1, False, alpha, beta, deep_search)
            if current_eval > max_eval:
                max_eval = current_eval
                optimal_move = move
            alpha = max(alpha, current_eval)
            if beta <= alpha:
                break  # Stop exploring this branch
        return max_eval, optimal_move
    else:
        min_eval = inf
        for move in valid_moves:
            next_cards, next_player1, next_player2 = simulate_move(cards, player1, player2, move, player=2)
            current_eval, _ = minimax(next_cards, next_player1, next_player2, depth - 1, True, alpha, beta, deep_search)
            if current_eval < min_eval:
                min_eval = current_eval
                optimal_move = move
            beta = min(beta, current_eval)
            if beta <= alpha:
                break  # Stop exploring this branch
        return min_eval, optimal_move


def simulate_move(cards, player1, player2, move, player):
    """
    Simulates a player's move and returns the resulting game state.

    Parameters:
        cards (list): A list of `Card` objects representing the current game state.
        player1 (Player): The Player 1 object including banners and cards.
        player2 (Player): The Player 2 object including banners and cards.
        move (int): The index of the card to move.
        player (int): The player making the move (1 for Player 1, 2 for Player 2).

    Returns:
        tuple: A tuple containing:
            - list: A deep copy of the updated `cards` after the move.
            - Player: A deep copy of the updated `player1` object.
            - Player: A deep copy of the updated `player2` object.
    """
    new_cards, new_player1, new_player2 = map(copy.deepcopy, [cards, player1, player2])
    current_player = new_player1 if player == 1 else new_player2
    selected_house = make_move(new_cards, move, current_player)
    set_banners(new_player1, new_player2, selected_house, player)
    return new_cards, new_player1, new_player2

def get_move(cards, player1, player2):
    """
    Determines the best move for Player 1 using the Minimax algorithm with Alpha-Beta pruning.

    Parameters:
        cards (list): A list of `Card` objects representing the current game state.
        player1 (Player): The Player 1 object including banners and cards.
        player2 (Player): The Player 2 object including banners and cards.

    Returns:
        int or None: The index of the best move for Player 1, or None if no moves are available.
    """
    depth = 9 if len(cards) < 25 else 5
    flag = True if (len(cards) <= 16) else False
    _, best_move = minimax(
        cards, player1, player2,
        depth=depth,
        is_maximizing=True,
        alpha=-inf,
        beta=inf,
        deep_search = flag
    )
    return best_move
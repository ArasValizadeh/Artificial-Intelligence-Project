from os import name as os_name
from os import system as os_system
from os.path import abspath, join, dirname
import copy
from main import make_move, set_banners,calculate_winner
from math import inf
import random

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

def get_player_moves(player_cards_dict, cards):
    """
    Gets the possible moves for a player based on their cards and the current board.

    Parameters:
        player_cards_dict (dict): Dictionary of player's cards grouped by house.
        cards (list): List of all Card objects on the board.

    Returns:
        list: List of possible moves (locations) for the player's cards.
    """
    moves = []

    # Flatten the player's cards dictionary into a list of Card objects
    player_cards = [card for card_list in player_cards_dict.values() for card in card_list]

    # Get the location of Varys
    varys_location = find_varys(cards)
    if varys_location is None:
        # If Varys is not on the board, no moves are possible
        return moves

    # Get the row and column of Varys
    varys_row, varys_col = varys_location // 6, varys_location % 6

    # Iterate through the player's cards and find valid moves
    for card in player_cards:
        row, col = card.get_location() // 6, card.get_location() % 6
        if row == varys_row or col == varys_col:
            moves.append(card.get_location())

    return moves

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
        'Stark': 9,'Greyjoy': 7,'Lannister': 7,'Targaryen': 6,'Baratheon': 5,'Tyrell': 9,'Tully': 10
    }
    banner_cards = {
        'Stark': 8,'Greyjoy': 7,'Lannister': 6,'Targaryen': 5,'Baratheon': 4,'Tyrell': 3,'Tully': 2
    }
    
    score = 0

    #calculate score for player 1
    for house in player1.get_banners():
        if player1.get_banners()[house]:
            score += banner_weights[house] * 10
            num_cards = 0
            for card in player1.get_cards():
                if card == house:
                    num_cards += 1
            if num_cards <= banner_cards[house] // 2 + 1:
                score += num_cards * 2

    # Calculate score for Player 2
    for house in player2.get_banners():
        if player2.get_banners()[house]:
            score -= banner_weights[house] * 10
            num_cards = 0
            for card in player1.get_cards():
                if card == house:
                    num_cards += 1
            if num_cards > banner_cards[house] // 2 + 1:
                penalty = -(banner_cards[house] // 2)
            else:
                penalty = num_cards
            score -= penalty * 2

    player1_moves = len(get_player_moves(player1.get_cards(), cards))
    player2_moves = len(get_player_moves(player2.get_cards(), cards))
    score += (player1_moves - player2_moves) * 3 # Reward more move options


    return score

def minimax(cards, player1, player2, depth, is_maximizing, alpha=-inf, beta=inf, no_heuristic=False):
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
        no_heuristic (bool, optional): Flag for additional end-game evaluation when no moves remain.

    Returns:
        tuple: Contains:
            - int: The highest or lowest score achievable from this state.
            - int or None: The index of the optimal move, or None if no valid moves are available.
    """
    valid_moves = get_valid_moves(cards)

    if depth == 0 or len(valid_moves) == 0:
        if no_heuristic:
            if calculate_winner(player1, player2) == 1:
                winner_score = 2000
            else:
                winner_score = -2000
            return (winner_score, None)
        else:
            return (evaluate_state(cards, player1, player2), None)

    optimal_move = None
    if is_maximizing:
        max_eval = -inf
        for move in valid_moves:
            next_cards, next_player1, next_player2 = simulate_move(cards, player1, player2, move, player=1)
            current_eval, _ = minimax(next_cards, next_player1, next_player2, depth - 1, False, alpha, beta, no_heuristic)
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
            current_eval, _ = minimax(next_cards, next_player1, next_player2, depth - 1, True, alpha, beta, no_heuristic)
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
    """
    # Create new instances instead of using deepcopy
    new_cards = []
    for card in cards:
        new_card = Card(card.get_house(), card.get_name(), card.get_location())
        new_cards.append(new_card)
    
    new_player1 = Player(player1.get_agent())
    new_player2 = Player(player2.get_agent())
    
    # Copy the cards and banners manually
    for house in player1.get_cards():
        for card in player1.get_cards()[house]:
            new_card = Card(card.get_house(), card.get_name(), card.get_location())
            new_player1.add_card(new_card)
    
    for house in player2.get_cards():
        for card in player2.get_cards()[house]:
            new_card = Card(card.get_house(), card.get_name(), card.get_location())
            new_player2.add_card(new_card)
    
    # Copy banners
    for house in player1.get_banners():
        if player1.get_banners()[house]:
            new_player1.get_house_banner(house)
    
    for house in player2.get_banners():
        if player2.get_banners()[house]:
            new_player2.get_house_banner(house)
    
    current_player = new_player1 if player == 1 else new_player2
    selected_house = make_move(new_cards, move, current_player)
    set_banners(new_player1, new_player2, selected_house, player)
    
    return new_cards, new_player1, new_player2

def get_move(cards, player1, player2, companion_cards, choose_companion):
    """
    Determines the best move for Player 1, now supporting companion cards.
    """
    # Create copies of the game state before passing to minimax
    new_cards = []
    for card in cards:
        new_card = Card(card.get_house(), card.get_name(), card.get_location())
        new_cards.append(new_card)
    
    new_player1 = Player(player1.get_agent())
    new_player2 = Player(player2.get_agent())
    
    # Copy the cards and banners manually
    for house in player1.get_cards():
        for card in player1.get_cards()[house]:
            new_card = Card(card.get_house(), card.get_name(), card.get_location())
            new_player1.add_card(new_card)
    
    for house in player2.get_cards():
        for card in player2.get_cards()[house]:
            new_card = Card(card.get_house(), card.get_name(), card.get_location())
            new_player2.add_card(new_card)
    
    # Copy banners
    for house in player1.get_banners():
        if player1.get_banners()[house]:
            new_player1.get_house_banner(house)
    
    for house in player2.get_banners():
        if player2.get_banners()[house]:
            new_player2.get_house_banner(house)

    if choose_companion:
        # Handle companion card selection (existing code)
        if companion_cards:
            selected_companion = list(companion_cards.keys())[0]
            move = [selected_companion]
            choices = companion_cards[selected_companion]['Choice']

            if choices == 1:
                move.append(random.choice(get_valid_moves(new_cards)))
            elif choices == 2:
                valid_moves = get_valid_moves(new_cards)
                if len(valid_moves) >= 2:
                    move.extend(random.sample(valid_moves, 2))
                else:
                    move.extend(valid_moves)
            elif choices == 3:
                valid_moves = get_valid_moves(new_cards)
                if len(valid_moves) >= 2 and len(companion_cards) > 1:
                    move.extend(random.sample(valid_moves, 2))
                    move.append(random.choice([c for c in companion_cards.keys() if c != selected_companion]))
                else:
                    move.extend(valid_moves)
                    move.append(random.choice(list(companion_cards.keys())) if companion_cards else None)
            return move

    # Use minimax with the copied game state
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
import math
from classes import Card, Player

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

    # Add points for completed houses for player1
    for house, cards_count in player1.get_cards().items():
        if len(cards_count) == house_total_cards[house]:
            score += 40 # Reward for completing a house

    # Subtract points for completed houses for player2
    for house, cards_count in player2.get_cards().items():
        if len(cards_count) == house_total_cards[house]:
            score -= 40  # Penalty for opponent completing a house

    # Add points for each card captured in favor of player1
    for house, cards_count in player1.get_cards().items():
        score += len(cards_count) * 5

    # Subtract points for each card captured in favor of player2
    for house, cards_count in player2.get_cards().items():
        score -= len(cards_count) * 5
    
    return score

def simulate_move(cards, move, player1, player2, is_maximizing):
    '''
    Simulates a move and updates the state of the game.
    
    Parameters:
        cards (list): list of Card objects
        move (int): the move to simulate
        player1 (Player): the current player
        player2 (Player): the opponent
        is_maximizing (bool): whether the current player is maximizing or not
    
    Returns:
        simulated_cards (list): the updated list of Card objects
        simulated_player1 (Player): the updated player1
        simulated_player2 (Player): the updated player2
    '''
    # Create deep copies of the cards
    simulated_cards = [Card(card.get_house(), card.get_name(), card.get_location()) for card in cards]

    # Manually duplicate player1 and player2
    simulated_player1 = Player(player1.get_agent())
    simulated_player1.cards = {house: cards[:] for house, cards in player1.get_cards().items()}
    simulated_player1.banners = player1.get_banners().copy()

    simulated_player2 = Player(player2.get_agent())
    simulated_player2.cards = {house: cards[:] for house, cards in player2.get_cards().items()}
    simulated_player2.banners = player2.get_banners().copy()

    current_player = simulated_player1 if is_maximizing else simulated_player2

    # Simulate capturing cards
    varys_location = find_varys(simulated_cards)
    varys_row, varys_col = varys_location // 6, varys_location % 6

    # Move Varys to the new location
    varys_card = [card for card in simulated_cards if card.get_name() == 'Varys'][0]
    varys_card.set_location(move)

    # Determine row/column and capture cards
    move_row, move_col = move // 6, move % 6
    for card in simulated_cards:
        card_row, card_col = card.get_location() // 6, card.get_location() % 6

        if (card_row == varys_row and card_col <= move_col) or (card_col == varys_col and card_row <= move_row):
            if card.get_house() != "No House" and card.get_house() == varys_card.get_house():
                current_player.add_card(card)

    return simulated_cards, simulated_player1, simulated_player2


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
        move_score = minimax(simulated_cards, new_player1, new_player2, depth=7, is_maximizing=True, alpha=-math.inf, beta=math.inf)
        if move_score > best_score:
            best_score = move_score
            best_move = move

    return best_move
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

    varys_row, varys_col = varys_location // 6, varys_location % 6

    moves = []

    for card in cards:
        if card.get_name() == 'Varys':
            continue

        row, col = card.get_location() // 6, card.get_location() % 6

        if row == varys_row or col == varys_col:
            moves.append(card.get_location())

    return moves

def get_valid_ramsay(cards):
    '''
    This function gets the possible moves for Ramsay.

    Parameters:
        cards (list): list of Card objects
    
    Returns:
        moves (list): list of possible moves
    '''

    moves=[]

    for card in cards:
        moves.append(card.get_location())
    
    return moves

def get_valid_jon_sandor_jaqan(cards):
    '''
    This function gets the possible moves for Jon Snow, Sandor Clegane, and Jaqen H'ghar.

    Parameters:
        cards (list): list of Card objects
    
    Returns:
        moves (list): list of possible moves
    '''

    moves=[]

    for card in cards:
        if card.get_name() != 'Varys':
            moves.append(card.get_location())
    
    return moves

def get_move(cards, player1, player2, companion_cards, choose_companion):
    '''
    This function gets the move of the player.

    Parameters:
        cards (list): list of Card objects
        player1 (Player): the player
        player2 (Player): the opponent
        companion_cards (dict): dictionary of companion cards
        choose_companion (bool): flag to choose a companion card

    Returns:
        move (int/list): the move of the player
    '''

    if choose_companion:
        # Choose a random companion card if available
        if companion_cards:
            selected_companion = random.choice(list(companion_cards.keys())) # Randomly select a companion card
            move = [selected_companion] # Add the companion card to the move list
            choices = companion_cards[selected_companion]['Choice'] # Get the number of choices required by the companion card
            
            if choices == 1:  # For cards like Jon Snow
                move.append(random.choice(get_valid_jon_sandor_jaqan(cards)))
            
            elif choices == 2:  # For cards like Ramsay
                valid_moves = get_valid_ramsay(cards)

                if len(valid_moves) >= 2:
                    move.extend(random.sample(valid_moves, 2))
                
                else:
                    move.extend(valid_moves)  # If not enough moves, just use what's available
                
                
            elif choices == 3:  # Special case for Jaqen with an additional companion card selection
                valid_moves = get_valid_jon_sandor_jaqan(cards)

                if len(valid_moves) >= 2 and len(companion_cards) > 0:
                    move.extend(random.sample(valid_moves, 2))
                    move.append(random.choice(list(companion_cards.keys())))
                
                else:
                    # If there aren't enough moves or companion cards, just return what's possible
                    move.extend(valid_moves)
                    move.append(random.choice(list(companion_cards.keys())) if companion_cards else None)
        
            return move
        
        else:
            # If no companion cards are left, just return an empty list to signify no action
            return []
    
    else:
        # Normal move, choose from valid moves
        moves = get_valid_moves(cards)
        return random.choice(moves) if moves else None
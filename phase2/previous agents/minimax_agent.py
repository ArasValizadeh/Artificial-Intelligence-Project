import random
from time import sleep
from main import make_move , make_companion_move , house_card_count , remove_unusable_companion_cards
import copy
from utils.classes import *
from itertools import combinations
import sys

def find_varys(cards):
    """
    This function finds the location of Varys on the board.

    Parameters:
        cards (list): list of Card objects

    Returns:
        varys_location (int): location of Varys
    """

    varys = [card for card in cards if card.get_name() == "Varys"]

    varys_location = varys[0].get_location()

    return varys_location


def get_valid_moves(cards):
    """
    This function gets the possible moves for the player.

    Parameters:
        cards (list): list of Card objects

    Returns:
        moves (list): list of possible moves
    """

    # Get the location of Varys
    varys_location = find_varys(cards)

    # Get the row and column of Varys
    varys_row, varys_col = varys_location // 6, varys_location % 6

    moves = []

    # Get the cards in the same row or column as Varys
    for card in cards:
        if card.get_name() == "Varys":
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
    val, best_move = get_best_move(
        cards, player1, player2, player1, companion_cards, choose_companion, depth=0, max_depth=1, max_depth_companion=0)
    return best_move


def get_best_move(cards, player1, player2, player, companion_cards, choose_companion, depth, max_depth, max_depth_companion = 0):
    """
    getting the best move for 4 depth from now
    approximate d:
    d[4] : 14.6 s
    d[5] : 161 s
    player1 : maximizer
    player2 : minimizer
    """
    try:
        # Normal move, choose from valid moves     
        if depth > max_depth:

            return (
                get_huristics(
                    cards=cards,
                    player=player,
                    player1=player1,
                    player2=player2,
                    ended=False,
                ),
                None,
            )
    
        if choose_companion:   
            # Choose a random companion card if available
            if player == player1:
                ans = -1e8
            else:
                ans = 1e8
            best_move = None
            
            if companion_cards:
                for selected_companion in list(companion_cards.keys()):
                    move = [selected_companion] # Add the companion card to the move list
                    choices = companion_cards[selected_companion]['Choice'] # Get the number of choices required by the companion card     
                    # For each choice required by the companion card
                    if choices == 0:
                        ans , best_move = make_move_for_companion(cards , player, player1,
                                                player2, companion_cards,depth, move , ans , best_move, max_depth_companion)

                    if choices == 1:  # For cards like Jon Snow
                        for valid_move in get_valid_jon_sandor_jaqan(cards):
                            temp_move = copy.deepcopy(move)
                            temp_move.append(valid_move)
                            # sleep(0.5)
                            ans , best_move = make_move_for_companion(cards , player, player1,
                                            player2, companion_cards,depth, temp_move , ans , best_move, max_depth_companion)
                            del temp_move
                    
                    elif choices == 2:  # For cards like Ramsay

                        valid_moves = get_valid_ramsay(cards)

                        if len(valid_moves) >= 2:
                            for valid_move  in list(combinations(valid_moves, 2)):
                                temp_move = copy.deepcopy(move)
                                temp_move.extend([valid_move[0] , valid_move[1]])
                                ans , best_move = make_move_for_companion(cards , player, player1,
                                            player2, companion_cards,depth, temp_move , ans , best_move, max_depth_companion)
                                del temp_move
                        
                        else:
                            temp_move = copy.deepcopy(move)
                            temp_move.extend(valid_moves)  # If not enough moves, just use what's available
                            ans , best_move = make_move_for_companion(cards , player, player1,
                                            player2, companion_cards,depth, temp_move , ans , best_move, max_depth_companion)
                            del temp_move
                        
                    elif choices == 3:  # Special case for Jaqen with an additional companion card selection
                        # continue
                        valid_moves = get_valid_jon_sandor_jaqan(cards)

                        if len(valid_moves) >= 2 and len(companion_cards) -1 > 0:
                            for v  in list(combinations(valid_moves, 2)):
                                for companion_choice in list(companion_cards.keys() - ["Jaqen"]):
                                    temp_move = copy.deepcopy(move)
                                    temp_move.extend([v[0] ,v[1] ,companion_choice])
                                    ans , best_move = make_move_for_companion(cards , player, player1,
                                            player2, companion_cards,depth, temp_move , ans , best_move, max_depth_companion)
                                    del temp_move
                        else:
                            # If there aren't enough moves or companion cards, just return what's possible
                            move.extend(valid_moves)
                            if companion_cards:
                                for companion_choice in list(companion_cards.keys()):
                                        temp_move = copy.deepcopy(move)
                                        temp_move.extend([companion_choice])
                                        ans , best_move = make_move_for_companion(cards , player, player1,
                                                player2, companion_cards,depth, temp_move , ans , best_move, max_depth_companion)
                                        del temp_move
                            else:
                                temp_move = copy.deepcopy(move)
                                temp_move.append([])
                                ans , best_move = make_move_for_companion(cards , player, player1,
                                            player2,companion_cards,depth, temp_move , ans , best_move, max_depth_companion) 
                                del temp_move                   
                

                return ans , best_move
            else:
                # If no companion cards are left, just return an empty list to signify no action
                ans , best_move = make_move_for_companion(cards , player, player1,
                                            player2, companion_cards, [] , ans , best_move, max_depth_companion) 
            return ans , best_move
        
        else:
            if player == player1:
                # maximizer player
                ans = -1e8
                best_move = None
                valid_moves = get_valid_moves(cards)
                if len(valid_moves) == 0:
                    return get_huristics(
                        cards=cards,
                        player=player,
                        player1=player1,
                        player2=player2,
                        ended=True,
                    ), None
                for move in valid_moves:
                    temp_cards = copy.deepcopy(cards)
                    temp_player1 = copy.deepcopy(player1)
                    temp_player2 = copy.deepcopy(player2)
                    temp_companion_cards = copy.deepcopy(companion_cards)
                    selected_house = make_move(cards=temp_cards, move=move,
                            player=temp_player1)
                    # determining if we have to choose companions or not:
                    next_depth = depth+1
                    next_choose_companion = False
                    next_player = temp_player2
                    if house_card_count(temp_cards, selected_house) == 0 and len(temp_companion_cards) != 0:
                        next_choose_companion = True
                        next_depth = depth
                        next_player = temp_player1
                    h_move, _ = get_best_move(
                        cards=temp_cards,
                        player1=temp_player1,
                        player2=temp_player2,
                        player=next_player,
                        companion_cards=temp_companion_cards,
                        choose_companion=next_choose_companion,
                        depth=next_depth,
                        max_depth=max_depth, 
                        max_depth_companion=max_depth_companion
                    )
                    del temp_cards
                    del temp_player1
                    del temp_player2
                    del temp_companion_cards
                    if ans < h_move:
                        ans, best_move = h_move, move
                return ans, best_move
            else:
                # minimizer player
                ans = 1e8
                best_move = None
                valid_moves = get_valid_moves(cards)
                if len(valid_moves) == 0:
                    return get_huristics(
                        cards=cards,
                        player=player,
                        player1=player1,
                        player2=player2,
                        ended=True,
                    ), None
                for move in valid_moves:
                    temp_cards = copy.deepcopy(cards)
                    temp_player1 = copy.deepcopy(player1)
                    temp_player2 = copy.deepcopy(player2)
                    temp_companion_cards = copy.deepcopy(companion_cards)
                    selected_house = make_move(cards=temp_cards, move=move,
                            player=temp_player2)
                    next_depth = depth+1
                    next_choose_companion = False
                    next_player = temp_player1
                    if house_card_count(temp_cards, selected_house) == 0 and len(temp_companion_cards) != 0:
                        next_choose_companion = True 
                        next_depth = depth 
                        next_player = temp_player2
                    h_move, _ = get_best_move(
                        cards=temp_cards,
                        player1=temp_player1,
                        player2=temp_player2,
                        player=next_player,
                        companion_cards=temp_companion_cards,
                        choose_companion=next_choose_companion,
                        depth=next_depth,
                        max_depth=max_depth,
                        max_depth_companion=max_depth_companion
                    )
                    del temp_cards
                    del temp_player1
                    del temp_player2
                    del temp_companion_cards
                    if ans > h_move:
                        ans, best_move = h_move, move
                return ans, best_move
    except KeyboardInterrupt:
        sys.exit()


def make_move_for_companion(cards , 
                        player , 
                        player1 , 
                        player2, 
                        companion_cards ,
                        depth,
                        move, 
                        ans,
                        best_move, 
                        max_depth_companion):
    temp_cards = copy.deepcopy(cards)
    temp_player = copy.deepcopy(player)
    temp_player1 = copy.deepcopy(player1)
    temp_player2 = copy.deepcopy(player2)
    temp_companion_cards = copy.deepcopy(companion_cards)
    temp_move = copy.deepcopy(move)
    make_companion_move(temp_cards , 
                        temp_companion_cards , 
                        temp_move , 
                        temp_player)
    
    if temp_move[0] == "Melisandre":
        next_player = temp_player
        next_depth = depth
    else:
        next_player = temp_player1 if temp_player == temp_player2 else temp_player2
        next_depth = depth + 1
    h_move , _ = get_best_move(
                temp_cards,
                temp_player1,
                temp_player2,
                next_player,
                temp_companion_cards,
                choose_companion=False,
                depth=next_depth,
                max_depth=max_depth_companion,
                max_depth_companion=max_depth_companion)
    del temp_cards
    del temp_player
    del temp_player1
    del temp_player2
    del temp_companion_cards
    if player == player1 and ans < h_move:
        ans , best_move = h_move , temp_move
    elif player == player2 and ans > h_move:
        ans , best_move = h_move,  temp_move
    del temp_move
    return ans , best_move


def get_huristics(cards, player: Player, player1: Player, player2: Player, ended):
    """
    finding the huristics for the given situation and the player
    """
    # {'Stark': [8], 'Greyjoy': [7], 'Lannister': [6], 'Targaryen': [5], 'Baratheon': [4], 'Tyrell': [3], 'Tully': [2]}
    # for player 1
    Stark1 = len(player1.cards["Stark"])
    Greyjoy1 = len(player1.cards["Greyjoy"])
    Lannister1 = len(player1.cards["Lannister"])
    Targaryen1 = len(player1.cards["Targaryen"])
    Baratheon1 = len(player1.cards["Baratheon"])
    Tyrell1 = len(player1.cards["Tyrell"])
    Tully1 = len(player1.cards["Tully"])
    # for player 2
    Stark2 = len(player2.cards["Stark"])
    Greyjoy2 = len(player2.cards["Greyjoy"])
    Lannister2 = len(player2.cards["Lannister"])
    Targaryen2 = len(player2.cards["Targaryen"])
    Baratheon2 = len(player2.cards["Baratheon"])
    Tyrell2 = len(player2.cards["Tyrell"])
    Tully2 = len(player2.cards["Tully"])
    return (Stark1 - Stark2 + 
            Greyjoy1 - Greyjoy2 +
            Lannister1 - Lannister2 + 
            Targaryen1 - Targaryen2 + 
            Baratheon1   - Baratheon2 +
            Tyrell1 - Tyrell2 +
            Tully1 - Tully2)
    stark_sum = Stark1 + Stark2
    greyjoy_sum = Greyjoy1 + Greyjoy2
    lannister_sum = Lannister1 + Lannister2
    targaryen_sum = Targaryen1 + Targaryen2
    baratheon_sum = Baratheon1 + Baratheon2
    tyrell_sum = Tyrell1 + Tyrell2
    tully_sum = Tully1 + Tully2

    p1score = 0
    p2score = 0
    win_points = 10
    diif_mull = 1.2

    if ended == True:
        if Stark1 > Stark2 or (Stark1 == Stark2 and player1.last["Stark"] == 1):
            p1score += win_points
        elif Stark2 > Stark1 or (Stark2 == Stark2 and player2.last["Stark"] == 1):
            p2score += win_points
        if Greyjoy1 > Greyjoy2 or (Greyjoy1 == Greyjoy2 and player1.last["Greyjoy"] == 1):
            p1score += win_points
        elif Greyjoy2 > Greyjoy1 or (Greyjoy2 == Greyjoy1 and player2.last["Greyjoy"] == 1):
            p2score += win_points
        if Lannister1 > Lannister2 or (Lannister1 == Lannister2 and player1.last["Lannister"] == 1):
            p1score += win_points
        elif Lannister2 > Lannister1 or (Lannister2 == Lannister1 and player2.last["Lannister"] == 1):
            p2score += win_points
        if Targaryen1 > Targaryen2 or (Targaryen1 == Targaryen2 and player1.last["Targaryen"] == 1):
            p1score += win_points
        elif Targaryen2 > Targaryen1 or (Targaryen2 == Targaryen1 and player2.last["Targaryen"] == 1):
            p2score += win_points
        if Baratheon1 > Baratheon2 or (Baratheon1 == Baratheon2 and player1.last["Baratheon"] == 1):
            p1score += win_points
        elif Baratheon2 > Baratheon1 or (Baratheon2 == Baratheon1 and player2.last["Baratheon"] == 1):
            p2score += win_points
        if Tyrell1 > Tyrell2 or (Tyrell1 == Tyrell2 and player1.last["Tyrell"] == 1):
            p1score += win_points
        elif Tyrell2 > Tyrell1 or (Tyrell2 == Tyrell1 and player2.last["Tyrell"] == 1):
            p2score += win_points
        if Tully1 > Tully2 or (Tully1 == Tully2 and player1.last["Tully"] == 1):
            p1score += win_points
        elif Tully2 > Tully1 or (Tully2 == Tully1 and player2.last["Tully"] == 1):
            p2score += win_points

        if p1score == p2score and player1.last["Stark"] == 1:
            p1score += 1
        elif p1score == p2score and player2.last["Stark"] == 1:
            p2score += 1
        # if p1score > p2score:
        #     print(p1score - p2score)
        if p1score > p2score:
            return 80
        else:
            return -80

    def f(num):
        return (num * (num + 1)) / 2

    # tully hue
    if tully_sum == 2:
        if player1.last["Tully"] == 1:
            p1score += win_points
        else:
            p2score += win_points

    # tyrell hue
    if tyrell_sum == 3:
        if Tyrell1 == 2:
            p1score += win_points
        else:
            p2score += win_points
    else:
        p1score += max(0, diif_mull * (f(Tyrell1) * 2) - f(Tyrell2) * 2)
        p2score += max(0, diif_mull * (f(Tyrell2) * 2) - f(Tyrell1) * 2)

    # baratheon hue
    if baratheon_sum == 4 or Baratheon1 > 2 or Baratheon2 > 2:
        if Baratheon1 > 2 or (Baratheon1 == 2 and player1.last["Baratheon"] == 1):
            p1score += win_points
        else:
            p2score += win_points
    elif baratheon_sum < 3:
        p1score += max(0, diif_mull * (f(Baratheon1)) - f(Baratheon2))
        p2score += max(0, diif_mull * (f(Baratheon2)) - f(Baratheon1))

    # targaryen hue
    if targaryen_sum == 5 or Targaryen1 > 2 or Targaryen2 > 2:
        if Targaryen1 > 2:
            p1score += win_points
        else:
            p2score += win_points
    else:
        p1score += max(0, diif_mull * (f(Targaryen1) / 3) - f(Targaryen2) / 3)
        p2score += max(0, diif_mull * (f(Targaryen2) / 3) - f(Targaryen1) / 3)
    # lannister hue
    if lannister_sum == 6 or Lannister1 > 3 or Lannister2 > 3:
        if Lannister1 > 3 or (Lannister1 == 3 and player1.last["Lannister"] == 1):
            p1score += win_points
        else:
            p2score += win_points
    elif lannister_sum < 5:
        p1score += max(0, diif_mull * (f(Lannister1) / 4) - f(Lannister2) / 4)
        p2score += max(0, diif_mull * (f(Lannister2) / 4) - Lannister1 / 4)

    # greyjoy hue
    if greyjoy_sum == 7 or Greyjoy1 > 3 or Greyjoy2 > 3:
        if Greyjoy1 > 3:
            p1score += win_points
        else:
            p2score += win_points
    else:
        p1score += max(0, diif_mull * (f(Greyjoy1) / 4) - f(Greyjoy2) / 4)
        p2score += max(0, diif_mull * (f(Greyjoy2) / 4) - f(Greyjoy1) / 4)

    # stark hue
    if stark_sum == 8 or Stark1 > 4 or Stark2 > 4:
        if Stark1 > 4 or (Stark1 == 4 and player1.last["Stark"] == 1):
            p1score += win_points * 1.001
        else:
            p2score += win_points * 1.001
    elif stark_sum < 7:
        p1score += max(0, diif_mull * (f(Stark1) / 4) - f(Stark2) / 4)
        p2score += max(0, diif_mull * (f(Stark2) / 4) - f(Stark1) / 4)
    # if p1score > p2score:
    #     print(p1score - p2score)
    return p1score - p2score
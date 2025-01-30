import torch
import torch.nn 
from utils.classes import Card, Player

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


def representation(full_cards: Card , player1s:Player , player2s:Player, companion_cards:Card):
    map_house = {'Stark': 1, 'Greyjoy': 2, 'Lannister': 3, 'Targaryen': 4, 'Baratheon': 5, 'Tyrell': 6, 'Tully': 7}
    map_companion_cards = {"Jon": 0,"Jaqen": 1,"Gendry": 2,"Melisandre": 3,"Ramsay": 4,"Sandor": 5}
    # making the representation for the cards
    ans = []
    for i in range(len(full_cards)):
        representation = torch.zeros((56))
        
        cards = full_cards[i]
        # print(cards)
        for card in  cards:
            if card.name == "Varys":
                representation[card.location] = -1
                continue
            house_label = map_house[card.get_house()]
            house_location = card.location
            representation[house_location] = house_label
        # making the representation for the companion_cards
        cards = companion_cards[i]
        for card in cards:
            representation[36+map_companion_cards[card]] = 1
        # making the representation for the player1
        player1 = player1s[i]
        for key in map_house.keys():
            representation[41 + map_house[key]] = len(player1.cards[key])
        # making the representation for the player2
        player2 = player2s[i]
        for key in map_house.keys():
            representation[48 + map_house[key]] = len(player2.cards[key])
        ans.append(representation)
    return ans
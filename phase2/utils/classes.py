class Card:
    '''
    This class represents a card in the game.
    '''

    def __init__(self, house, name, location):
        '''
        This function initializes the card.

        Parameters:
            house (str): the house of the card
            name (str): the name of the card
            location (int): the location of the card
        '''

        self.house = house
        self.name = name
        self.location = location
    
    def get_house(self):
        '''
        This function returns the house of the card.

        Returns:
            house (str): the house of the card
        '''

        return self.house
    
    def get_name(self):
        '''
        This function returns the name of the card.

        Returns:
            name (str): the name of the card
        '''

        return self.name
    
    def get_location(self):
        '''
        This function returns the location of the card.

        Returns:
            location (int): the location of the card
        '''
        return self.location
    
    def set_location(self, location):
        '''
        This function sets the location of the card.

        Parameters:
            location (int): the new location of the card
        '''

        self.location = location

class Player:
    '''
    This class represents a player in the game.
    '''

    def __init__(self, agent):
        '''
        This function initializes the player.

        Parameters:
            agent (str): the agent of the player
        '''

        self.agent = agent
        self.cards = {'Stark': [], 'Greyjoy': [], 'Lannister': [], 'Targaryen': [], 'Baratheon': [], 'Tyrell': [], 'Tully': []}
        self.banners = {'Stark': 0, 'Greyjoy': 0, 'Lannister': 0, 'Targaryen': 0, 'Baratheon': 0, 'Tyrell': 0, 'Tully': 0}

    def get_agent(self):
        '''
        This function returns the agent of the player.

        Returns:
            agent (str): the agent of the player
        '''

        return self.agent
    
    def get_cards(self):
        '''
        This function returns the cards of the player.

        Returns:
            cards (dict): the cards of the player
        '''

        return self.cards
    
    def get_banners(self):
        '''
        This function returns the banners of the player.

        Returns:
            banners (dict): the banners of the player
        '''

        return self.banners
    
    def add_card(self, card):
        '''
        This function adds a card to the player.

        Parameters:
            card (Card): the card to add to the player
        '''

        self.cards[card.get_house()].append(card)
    
    def get_house_banner(self, house):
        '''
        This function gives the banner of a house to the player.

        Parameters:
            house (str): the house to give the banner to the player
        '''

        self.banners[house] = 1
    
    def remove_house_banner(self, house):
        '''
        This function removes the banner of a house from the player.

        Parameters:
            house (str): the house to remove the banner from the player
        '''

        self.banners[house] = 0
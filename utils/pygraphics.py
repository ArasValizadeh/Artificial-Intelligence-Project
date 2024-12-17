import pygame
import json
import time
from os import pardir, environ
from os.path import abspath, join, dirname

# Get the path of the assets folder
assets_path = join((abspath(join(dirname(abspath(__file__)), pardir))), "assets")

ROWS = 6 # Number of rows in the board
COLS = 6 # Number of columns in the board
CARD_SIZE = 110  # Size of the card
MARGIN = 15  # Space in between cards
FOOTER_SIZE = 30  # Size of the footer
BOARD_HEIGHT = ROWS * CARD_SIZE + (ROWS - 1) * MARGIN + FOOTER_SIZE  # Height of the board
BOARD_WIDTH = COLS * CARD_SIZE + (COLS - 1) * MARGIN  # Width of the board
WINNER_HEIGHT_OFFSET = 36  # Height offset of the winner text
assets = {} # Dictionary to store every asset

def load_assets():
    '''
    This function loads the assets of the game.
    '''

    # Get the characters of the game
    with open(join(assets_path, 'characters.json')) as f:
        characters = json.load(f)
    house_characters = list(characters.values())

    # Load all the images of the cards
    for house in house_characters:
        for character in house:
            # Load the image of the card
            assets[character] = pygame.image.load(join(assets_path, 'cards', character + ".jpg"))

            # Resize the image of the card
            assets[character] = pygame.transform.scale(assets[character], (CARD_SIZE, CARD_SIZE))
    
    # Load the icon of the window
    assets['icon'] = pygame.image.load(join(assets_path, 'icons', 'icon.jpg'))

    # Resize the icon of the window
    assets['icon'] = pygame.transform.scale(assets['icon'], (256, 256))

    # Set the font of the text (Arial, 20pt)
    font = pygame.font.SysFont('Arial', 20)

    # Render the texts
    assets['0'] = font.render('A Game of Thrones: Hand of the King', True, [0, 0, 0])
    assets['1'] = font.render('Player 1\'s turn', True, [0, 0, 0])
    assets['2'] = font.render('Player 2\'s turn', True, [0, 0, 0])

    # Load the background of win screen
    assets['win_screen'] = pygame.image.load(join(assets_path, 'backgrounds', 'win_screen.jpg'))

    # Resize the background of win screen to the size of the board
    assets['win_screen'] = pygame.transform.scale(assets['win_screen'], (BOARD_WIDTH, BOARD_HEIGHT))

def init_board():
    '''
    This function initializes the board.

    Returns:
        screen (pygame.Surface): the screen for the game
    '''

    # Initialize Pygame
    pygame.init()

    # Get the size of the monitor
    monitor_info = pygame.display.Info()

    # Check if the board fits the monitor
    global BOARD_HEIGHT, BOARD_WIDTH
    if BOARD_WIDTH > monitor_info.current_w or BOARD_HEIGHT > monitor_info.current_h:
        # Change the size of the cards
        global CARD_SIZE
        CARD_SIZE = 90

        # Change the size of the board to fit the monitor
        BOARD_HEIGHT = ROWS * CARD_SIZE + (ROWS - 1) * MARGIN + FOOTER_SIZE
        BOARD_WIDTH = COLS * CARD_SIZE + (COLS - 1) * MARGIN

        # Change the winner height offset
        global WINNER_HEIGHT_OFFSET
        WINNER_HEIGHT_OFFSET = 31

        # Put the board in the top center of the screen
        environ.pop('SDL_VIDEO_CENTERED', None) # Remove the previous setting
        environ['SDL_VIDEO_WINDOW_POS'] = '%d, 30' % ((monitor_info.current_w - BOARD_WIDTH) // 2)
    
    else:
        # Put the board in the center of the screen
        environ['SDL_VIDEO_CENTERED'] = '1'

    # Load the assets of the game
    load_assets()

    # Set the title of the window
    pygame.display.set_caption('A Game of Thrones: Hand of the King')

    # Set the icon of the window
    icon = assets['icon']
    pygame.display.set_icon(icon)

    # Set the size of the board
    board = pygame.display.set_mode([BOARD_WIDTH, BOARD_HEIGHT])

    # Set the background color of the board to white
    board.fill([255, 255, 255])

    return board

def update():
    '''
    This function updates the display.
    '''

    pygame.display.update()

def draw_footer(board, text):
    '''
    This function draws the footer on the board.

    Parameters:
        text (str): text to display in the footer
    '''

    # Get the text to display
    text = assets[text]

    # Get the size of the text
    text_rect = text.get_rect()

    # Set the position of the text in the center of the footer
    text_rect.center = (BOARD_WIDTH // 2, BOARD_HEIGHT - FOOTER_SIZE // 2)

    # Draw the text on the board
    board.blit(text, text_rect)

def draw_board(board, cards, banner_footer):
    '''
    This function draws the cards on the board.

    Parameters:
        board (pygame.Surface): the screen for the game
        cards (list): list of Card objects
        banner_footer (str): text to display in the footer
    '''

    # Clear the board
    board.fill([255, 255, 255])

    for card in cards:
        # Get the location of the card
        location = card.get_location()

        # Calculate the row and column of the card
        row, col = location // COLS, location % COLS 

        # Calculate the position of the card
        x = col * CARD_SIZE + col * MARGIN
        y = row * CARD_SIZE + row * MARGIN

        # Load the image of the card
        card_img = assets[card.get_name()]

        # Draw the card on the board
        board.blit(card_img, (x, y))
    
    # Draw the footer
    draw_footer(board, banner_footer)

    # Update the display
    update()

def display_winner(board, winner, winner_agent):
    '''
    This function displays the winner of the game.

    Parameters:
        board (pygame.Surface): the screen for the game
        winner (int): the number of the winner
        winner_agent (str): the agent of the winner
    '''

    # Clear the board
    board.fill([255, 255, 255])

    # Load the background of the win screen
    win_screen = assets['win_screen']

    # Draw the background of the win screen
    board.blit(win_screen, (0, 0))

    # Set the font of the text (Arial, 25pt)
    font = pygame.font.SysFont('Arial', 25)

    # Get the text to display
    if winner_agent == 'human':
        text = 'Player ' + str(winner)
    
    else:
        text = winner_agent[max(0, winner_agent.find('/'), winner_agent.find('\\')):]

    # Render the text
    text = font.render(text + ' wins!', True, [255, 255, 255])

    # Get the size of the text
    text_rect = text.get_rect()

    # Set the position of the text in the center of the board
    text_rect.center = (BOARD_WIDTH // 2 - 12, BOARD_HEIGHT // 2 + WINNER_HEIGHT_OFFSET)

    # Draw the text on the board
    board.blit(text, text_rect)

    # Update the display
    update()

def show_board(seconds):
    '''
    This function shows the board for a certain amount of time.

    Parameters:
        seconds (int): number of seconds to show the board
    '''

    # Get the initial time
    initial_time = time.time()

    # Show the board for the given amount of time
    while time.time() - initial_time < seconds:
        for event in pygame.event.get():
            # Check if the event is the close button
            if event.type == pygame.QUIT:
                # Close the window
                pygame.quit()

                # Exit the program
                exit()

def get_player_move():
    '''
    This function gets the move of the player.

    Returns:
        location (int): location of the card
    '''

    # Check if the player has made a move
    move_made = False

    while not move_made:
        # Get the event
        for event in pygame.event.get():
            # Check if the event is a mouse click
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Get the position of the mouse
                x, y = pygame.mouse.get_pos()

                # Calculate the row and column of the card
                col = x // (CARD_SIZE + MARGIN)
                row = y // (CARD_SIZE + MARGIN)

                # Calculate the location of the card
                location = row * COLS + col

                # Check if the location is valid
                if location < ROWS * COLS:
                    move_made = True
            
            # Check if the event is the close button
            elif event.type == pygame.QUIT:
                # Close the window
                pygame.quit()

                # Exit the program
                exit()

    return location
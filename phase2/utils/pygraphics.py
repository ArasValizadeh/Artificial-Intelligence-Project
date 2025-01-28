import pygame
import json
import time
from moviepy.editor import ImageSequenceClip
from numpy import rot90, flipud
from os import pardir, environ
from os.path import abspath, join, dirname

# Get the path of the assets and videos folder
assets_path = join((abspath(join(dirname(abspath(__file__)), pardir))), "assets")
videos_path = join((abspath(join(dirname(abspath(__file__)), pardir))), "videos")

ROWS = 6 # Number of rows in the board
COLS = 6 # Number of columns in the board
CARD_SIZE = 110  # Size of the card
MARGIN = 15  # Space in between cards
FOOTER_SIZE = 30  # Size of the footer
BOARD_HEIGHT = ROWS * CARD_SIZE + (ROWS - 1) * MARGIN + FOOTER_SIZE  # Height of the board
BOARD_WIDTH = (COLS + 3) * CARD_SIZE + (COLS + 3 - 1) * MARGIN # Width of the board
WIN_WIDTH = COLS * CARD_SIZE + (COLS - 1) * MARGIN  # Width of the win screen
WINNER_HEIGHT_OFFSET = 36  # Height offset of the winner text
assets = {} # Dictionary to store every asset
frames = [] # List to store the frames of the video

def load_assets():
    '''
    This function loads the assets of the game.
    '''

    # Get the characters of the game
    with open(join(assets_path, 'characters.json')) as f:
        characters = json.load(f)
    
    companions = characters['Companion'] # Companions of the game

    # Remove the companions from the characters
    del characters['Companion']

    # Load the images of the companions
    for companion in companions:
        # Load the image of the companion
        assets[companion] = pygame.image.load(join(assets_path, 'companions', companion + ".jpg"))

        # Resize the image of the companion
        assets[companion] = pygame.transform.scale(assets[companion], (CARD_SIZE * 1.5, CARD_SIZE * 2.3))

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
    assets['CC'] = font.render('Choose a companion', True, [0, 0, 0])
    assets['BC1'] = font.render('Choose the first card', True, [0, 0, 0])
    assets['BC2'] = font.render('Choose the second card', True, [0, 0, 0])

    # Load the background of win screen
    assets['win_screen'] = pygame.image.load(join(assets_path, 'backgrounds', 'win_screen.jpg'))

    # Resize the background of win screen to the size of the board
    assets['win_screen'] = pygame.transform.scale(assets['win_screen'], (WIN_WIDTH, BOARD_HEIGHT))
    
    separation_x = COLS * CARD_SIZE + (COLS - 1) * MARGIN + (MARGIN // 2)
    gray_color = (128, 128, 128, 128)  # (R, G, B, Alpha)

    # Create a semi-transparent gray surface to cover the cards
    cards_gray_surface = pygame.Surface((separation_x, BOARD_HEIGHT - FOOTER_SIZE), pygame.SRCALPHA)
    cards_gray_surface.fill(gray_color)

    # Store the gray surface in the assets
    assets['cards_gray_surface'] = cards_gray_surface

    # Create a semi-transparent gray surface to cover the companions
    companions_gray_surface = pygame.Surface((BOARD_WIDTH - separation_x, BOARD_HEIGHT), pygame.SRCALPHA)
    companions_gray_surface.fill(gray_color)

    # Store the gray surface in the assets
    assets['companions_gray_surface'] = companions_gray_surface

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
        BOARD_WIDTH = (COLS + 3) * CARD_SIZE + (COLS + 3 - 1) * MARGIN

        global WIN_WIDTH
        WIN_WIDTH = COLS * CARD_SIZE + (COLS - 1) * MARGIN

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

def store_frame(board, needs_resize = False, FPS = 30):
    '''
    This function stores the frame of the board.

    Parameters:
        board (pygame.Surface): the screen for the game
        needs_resize (bool): whether the frame needs to be resized
        FPS (int): frames per second
    '''

    frame = pygame.surfarray.array3d(board) # Get the frame

    if needs_resize: # For the win screen
        frame = pygame.transform.smoothscale(pygame.surfarray.make_surface(frame), (BOARD_WIDTH, BOARD_HEIGHT)) # Resize the frame
        frame = pygame.surfarray.array3d(frame) # Get the frame

    frame = rot90(frame) # Rotate the frame
    frame = flipud(frame) # Flip the frame

    for _ in range(FPS):
        frames.append(frame) # Store the frame

def save_video(file_name):
    '''
    This function saves the video of the game.

    Parameters:
        file_name (str): name of the video file
    '''

    # Create a video of the game
    clip = ImageSequenceClip(frames, fps = 30)

    # Save the video
    clip.write_videofile(join(videos_path, file_name + '.mp4'), codec = 'libx264')

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
    text_x = (COLS * CARD_SIZE + (COLS - 1) * MARGIN) // 2
    text_y = BOARD_HEIGHT - FOOTER_SIZE // 2
    text_rect.center = (text_x, text_y)

    # Draw the text on the board
    board.blit(text, text_rect)

def draw_companions(board, companions):
    '''
    This function draws the companions on the board.

    Parameters:
        board (pygame.Surface): the screen for the game
        companions (dict): dictionary of companions
    '''

    # Draw the companions on the board
    for companion in companions:
        row, col = companions[companion]['Row'], companions[companion]['Column']

        # Calculate the position of the companion
        x = col * CARD_SIZE + col * MARGIN
        y = row * CARD_SIZE + row * MARGIN + 5 * (row // 2)

        # Load the image of the companion
        companion_img = assets[companion]

        # Draw the companion on the board
        board.blit(companion_img, (x, y))

def draw_board(board, cards, companions, banner_footer, is_cards_gray = False):
    '''
    This function draws the cards on the board.

    Parameters:
        board (pygame.Surface): the screen for the game
        cards (list): list of Card objects
        companions (dict): dictionary of companions
        banner_footer (str): text to display in the footer
        is_cards_gray (bool): whether the cards should be grayed out
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

    # Draw a horizontal line to separate the companions from the cards
    line_x = COLS * CARD_SIZE + (COLS - 1) * MARGIN + (MARGIN // 2)
    pygame.draw.line(board, [0, 0, 0], (line_x, 0), (line_x, BOARD_HEIGHT), 2)

    # Draw the companions
    draw_companions(board, companions)

    if is_cards_gray: # True means companions must be chosen
        # Get the gray surface to cover the cards
        gray_surface = assets['cards_gray_surface']

        # Draw the gray surface on the board
        board.blit(gray_surface, (0, 0))
    
    elif is_cards_gray is not None:
        # Get the gray surface to cover the companions
        gray_surface = assets['companions_gray_surface']

        # Draw the gray surface on the board
        board.blit(gray_surface, (line_x, 0))
    
    else: # None means both cards and companions must be grayed out
        # Get the gray surface to cover the cards
        gray_surface = assets['cards_gray_surface']

        # Draw the gray surface on the board
        board.blit(gray_surface, (0, 0))

        # Get the gray surface to cover the companions
        gray_surface = assets['companions_gray_surface']

        # Draw the gray surface on the board
        board.blit(gray_surface, (line_x, 0))

    # Update the display
    update()

    store_frame(board) # Store the frame

def display_winner(board, winner, winner_agent):
    '''
    This function displays the winner of the game.

    Parameters:
        board (pygame.Surface): the screen for the game
        winner (int): the number of the winner
        winner_agent (str): the agent of the winner
    '''

    board = pygame.display.set_mode([WIN_WIDTH, BOARD_HEIGHT])

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
    text_rect.center = (WIN_WIDTH // 2 - 12, BOARD_HEIGHT // 2 + WINNER_HEIGHT_OFFSET)

    # Draw the text on the board
    board.blit(text, text_rect)

    # Update the display
    update()

    store_frame(board, True) # Store the frame

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

def get_player_move(card_moves, companions = None):
    '''
    This function gets the move of the player.

    Parameters:
        card_moves (list): list of possible card moves
        companions (dict): dictionary of companions (If the player can choose a companion)

    Returns:
        location (int): location of the card
    '''

    # Check if the player can choose a companion
    if companions is not None:
        # Store the possible areas where the player can click
        possible_moves = []

        for companion in companions:
            # Get the row and column of the companion
            row = companions[companion]['Row']
            col = companions[companion]['Column']

            # Calculate the starting position of the companion
            start_x = col * CARD_SIZE + col * MARGIN
            start_y = row * CARD_SIZE + row * MARGIN + 5 * (row // 2)

            # Store the possible area where the player can click
            possible_moves.append([companion,
                                   (start_x, start_y),
                                   (start_x + CARD_SIZE * 1.5, start_y + CARD_SIZE * 2.3)])

    # Check if the player has made a move
    move_made = False

    while not move_made:
        # Get the event
        for event in pygame.event.get():
            # Check if the event is a mouse click
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Get the position of the mouse
                x, y = pygame.mouse.get_pos()

                # Check if the player clicked on a companion
                if companions is not None:
                    for companion in possible_moves:
                        if companion[1][0] <= x <= companion[2][0] and companion[1][1] <= y <= companion[2][1]:
                            location = [companion[0]] # Get the name of the companion
                            move_made = True
                            break
                    
                    if move_made: # Exit the loop if the player clicked on a companion
                        break

                    else:
                        continue # Should select a companion

                # Calculate the row and column of the card
                col = x // (CARD_SIZE + MARGIN)
                row = y // (CARD_SIZE + MARGIN)

                # Calculate the location of the card
                location = row * COLS + col

                # Check if the location is valid
                if location < ROWS * COLS and location in card_moves:
                    move_made = True
            
            # Check if the event is the close button
            elif event.type == pygame.QUIT:
                # Close the window
                pygame.quit()

                # Exit the program
                exit()

    return location

def close_board():
    '''
    This function closes the board.
    '''

    # Close the window
    pygame.quit()
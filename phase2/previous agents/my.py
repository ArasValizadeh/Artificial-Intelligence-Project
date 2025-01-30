import math
import copy
import random
import time
from utils.classes import Card, Player
from main import make_move


def find_varys(cards):
    """یافتن موقعیت واریس در صفحه بازی"""
    varys = [card for card in cards if card.get_name() == 'Varys']
    return varys[0].get_location()


def get_valid_moves(cards):
    """دریافت لیست حرکت‌های معتبر برای واریس"""
    varys_location = find_varys(cards)
    varys_row, varys_col = varys_location // 6, varys_location % 6
    moves = [card.get_location() for card in cards if card.get_name() != 'Varys' and
             (card.get_location() // 6 == varys_row or card.get_location() % 6 == varys_col)]
    return moves


def is_game_over(cards):
    """بررسی پایان بازی"""
    return len(get_valid_moves(cards)) == 0


def house_card_count(cards, house):
    """تعداد کارت‌های یک خاندان خاص در زمین"""
    return sum(1 for card in cards if card.get_house() == house)


def evaluate(cards, player1, player2, last_taken_house, last_turn):
    """ارزیابی وضعیت فعلی بازی"""
    player1_score = sum(player1.get_banners().values()) * 5
    player2_score = sum(player2.get_banners().values()) * 5

    player1_score += sum(len(player1.get_cards()[h]) for h in player1.get_cards()) * 1.5
    player2_score += sum(len(player2.get_cards()[h]) for h in player2.get_cards()) * 1.5

    player1_score += len(get_valid_moves(cards)) * 0.3
    player2_score += len(get_valid_moves(cards)) * 0.3

    if last_taken_house:
        if last_turn == 1:
            player1_score += 4
        else:
            player2_score += 4

    return player1_score - player2_score


def simulate_move(cards, player1, player2, move, current_player):
    """شبیه‌سازی حرکت و ایجاد حالت جدید بازی"""
    cards_copy, player1_copy, player2_copy = copy.deepcopy(cards), copy.deepcopy(player1), copy.deepcopy(player2)
    selected_house = make_move(cards_copy, move, player1_copy if current_player == 1 else player2_copy)
    return cards_copy, player1_copy, player2_copy, selected_house


def adaptive_minimax_depth(cards):
    """انتخاب عمق داینامیک برای Minimax بر اساس تعداد کارت‌های باقی‌مانده"""
    remaining_cards = len(cards)
    return 10 if remaining_cards < 12 else (7 if remaining_cards < 20 else 5)


def genetic_algorithm(cards, player1, player2, population_size=10, generations=5):
    """الگوریتم ژنتیک برای انتخاب بهترین حرکت اولیه"""
    population = [random.choice(get_valid_moves(cards)) for _ in range(population_size)]

    for _ in range(generations):
        scored_population = [(move, evaluate(*simulate_move(cards, player1, player2, move, 1)[:3], None, None))
                             for move in population]
        scored_population.sort(key=lambda x: x[1], reverse=True)
        population = [move for move, _ in scored_population[:population_size // 2]]
        population += [random.choice(population) for _ in range(population_size // 2)]

    return max(population, key=lambda move: evaluate(*simulate_move(cards, player1, player2, move, 1)[:3], None, None))


def minimax(cards, player1, player2, depth, maximizing, alpha, beta):
    """Minimax با هرس آلفا-بتا"""
    if depth == 0 or is_game_over(cards):
        return evaluate(cards, player1, player2, None, None), None

    valid_moves = get_valid_moves(cards)
    best_move = None

    if maximizing:
        max_eval = -math.inf
        for move in valid_moves:
            new_cards, new_player1, new_player2, _ = simulate_move(cards, player1, player2, move, 1)
            eval_value, _ = minimax(new_cards, new_player1, new_player2, depth - 1, False, alpha, beta)
            if eval_value > max_eval:
                max_eval, best_move = eval_value, move
            alpha = max(alpha, eval_value)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in valid_moves:
            new_cards, new_player1, new_player2, _ = simulate_move(cards, player1, player2, move, 2)
            eval_value, _ = minimax(new_cards, new_player1, new_player2, depth - 1, True, alpha, beta)
            if eval_value < min_eval:
                min_eval, best_move = eval_value, move
            beta = min(beta, eval_value)
            if beta <= alpha:
                break
        return min_eval, best_move


def get_companion_move(companion_cards, cards, player1, player2):
    """انتخاب بهترین کارت همراه، با مکانیزم صحیح `Jaqen`"""

    if not companion_cards:
        print("DEBUG: No companion cards left. Choosing random move.")
        return [random.choice(get_valid_moves(cards))]

    print(f"DEBUG: Available companions: {list(companion_cards.keys())}")

    best_move = None
    best_score = -math.inf
    selected_companion = None
    print(f"companion_cards: {companion_cards}")
    for companion in companion_cards:
        print(f"DEBUG: Evaluating companion: {companion}")
        move = [companion]  # شروع حرکت با نام کارت همراه
        move_eval = evaluate(cards, player1, player2, None, None)

        if companion == 'Melisandre':
            move_eval += 4

        elif companion == 'Jon':
            valid_targets = [card for card in cards if house_card_count(cards, card.get_house()) > 1]
            if valid_targets:
                target = max(valid_targets, key=lambda c: house_card_count(cards, c.get_house()))
                move.append(target.get_location())
                move_eval += 3
            else:
                continue

        elif companion == 'Gendry':
            move_eval += 2

        elif companion == 'Jaqen':
            valid_moves = [card for card in cards if card.get_name() != 'Varys']
            if len(valid_moves) >= 2 and len(companion_cards) > 1:
                chosen_cards = random.sample(valid_moves, 2)
                remaining_companions = [c for c in companion_cards if c != 'Jaqen']
                if remaining_companions:
                    chosen_companion = random.choice(remaining_companions)
                    move.extend([chosen_cards[0].get_location(), chosen_cards[1].get_location()])
                    move.append(chosen_companion)
                    move_eval += 4
                else:
                    print("DEBUG: No valid companion to remove for Jaqen")
                    continue
            else:
                print("DEBUG: Skipping Jaqen due to lack of valid targets.")
                continue

        elif companion == 'Sandor':
            valid_moves = [card for card in cards if card.get_name() != 'Varys']
            if valid_moves:
                move.append(random.choice(valid_moves).get_location())
                move_eval += 3
            else:
                continue

        elif companion == 'Ramsay':
            valid_moves = [card for card in cards]
            if len(valid_moves) >= 2:
                move.extend([random.choice(valid_moves).get_location(), random.choice(valid_moves).get_location()])
                move_eval += 2.5
            else:
                continue

        print(f"DEBUG: Evaluated {companion} with move {move} and score {move_eval}")

        if move_eval > best_score:
            best_score = move_eval
            best_move = move
            selected_companion = companion

    if best_move:
        print(f"DEBUG: Selected companion move: {best_move}")
        return best_move
    else:
        random_choice = [random.choice(list(companion_cards.keys()))]
        print(f"DEBUG: No good move found, selecting random companion: {random_choice}")
        return random_choice


def hybrid_move_selection(cards, player1, player2):
    """انتخاب ترکیبی حرکت: الگوریتم ژنتیک + Minimax"""
    start_time = time.time()
    candidate_moves = [genetic_algorithm(cards, player1, player2) for _ in range(5)]
    best_move, best_eval = None, -math.inf
    alpha, beta, depth = -math.inf, math.inf, adaptive_minimax_depth(cards)

    for move in candidate_moves:
        if time.time() - start_time > 5:
            break
        new_cards, new_player1, new_player2, _ = simulate_move(cards, player1, player2, move, 1)
        eval_value, _ = minimax(new_cards, new_player1, new_player2, depth, True, alpha, beta)
        if eval_value > best_eval:
            best_eval, best_move = eval_value, move

    return best_move if best_move else random.choice(get_valid_moves(cards))


def get_move(cards, player1, player2, companion_cards, choose_companion):
    """دریافت حرکت از ایجنت، شامل انتخاب کارت همراه در صورت لزوم"""

    print(f"DEBUG: Entering get_move with choose_companion={choose_companion}")

    if choose_companion:
        move = get_companion_move(companion_cards, cards, player1, player2)
        print(f"DEBUG: Choosing companion: {move}")  # لاگ برای بررسی خروجی

        if not move:
            print("DEBUG: get_companion_move() returned None, selecting random companion")
            move = [random.choice(list(companion_cards.keys()))]

        print(f"DEBUG: Selected companion move: {move}")
        return move

    move = hybrid_move_selection(cards, player1, player2)
    print(f"DEBUG: Choosing normal move: {move}")  # لاگ برای بررسی خروجی
    return move


def get_valid_ramsay(cards):
    """دریافت کارت‌هایی که رمزی اسنو می‌تواند جابه‌جا کند."""
    valid_moves = [card.get_location() for card in cards if card.get_name() != 'Varys']
    return valid_moves if len(valid_moves) >= 2 else []


def get_valid_jon_sandor_jaqan(cards):
    """دریافت کارت‌هایی که جان اسنو، سندور کلیگن و جکن هگار می‌توانند حذف کنند."""
    valid_moves = [card.get_location() for card in cards if card.get_name() != 'Varys']
    return valid_moves if valid_moves else []







def get_move(cards, player1, player2, companion_cards, choose_companion):
    """دریافت حرکت ایجنت، همراه با لاگ برای عیب‌یابی"""

    if choose_companion:
        move = get_companion_move(companion_cards, cards, player1, player2)
        print(f"Choosing companion: {move}")  # لاگ برای بررسی خروجی
        if not move:
            move = [random.choice(list(companion_cards.keys()))]  # اگر مقدار نامعتبر برگردد، کارت تصادفی انتخاب کن
        return move

    move = hybrid_move_selection(cards, player1, player2)
    print(f"Choosing normal move: {move}")  # لاگ برای بررسی خروجی
    return move


def make_companion_move(cards, companion_cards, move, player):
    """اجرای حرکت کارت همراه، مخصوصاً `Jaqen`، با حذف صحیح کارت‌ها"""

    if not move:
        print("DEBUG: No move received in make_companion_move()")
        return None

    selected_companion = move[0]
    print(f"DEBUG: Executing companion move for {selected_companion}")

    if selected_companion == 'Jon':
        selected_card = find_card(cards, move[1])
        if selected_card:
            house = selected_card.get_house()
            player.add_card(Card(house, 'Jon Snow', -1))
            player.add_card(Card(house, 'Jon Snow', -1))
            print(f"DEBUG: Jon Snow effect applied for house {house}")
        return selected_companion

    elif selected_companion == 'Gendry':
        player.add_card(Card('Baratheon', 'Gendry', -1))
        print("DEBUG: Gendry effect applied.")
        return 'Baratheon'

    elif selected_companion == 'Jaqen':
        if len(move) < 4:
            print("DEBUG: Invalid move for Jaqen, skipping.")
            return None

        first_card = find_card(cards, move[1])
        second_card = find_card(cards, move[2])
        selected_companion_card = move[3]

        print(f"DEBUG: Jaqen selected, removing cards at {move[1]} and {move[2]} and companion {move[3]}")

        if first_card and first_card in cards:
            print(f"DEBUG: Removing {first_card.get_name()} at {first_card.get_location()}")
            cards.remove(first_card)

        if second_card and second_card in cards:
            print(f"DEBUG: Removing {second_card.get_name()} at {second_card.get_location()}")
            cards.remove(second_card)

        if selected_companion_card in companion_cards:
            print(f"DEBUG: Removing companion card {selected_companion_card}")
            del companion_cards[selected_companion_card]

        if 'Jaqen' in companion_cards:
            print("DEBUG: Removing Jaqen from companion list")
            del companion_cards['Jaqen']

        print("DEBUG: Jaqen move executed successfully!")
        return None

    elif selected_companion == 'Sandor':
        selected_card = find_card(cards, move[1])
        if selected_card and selected_card in cards:
            print(f"DEBUG: Removing {selected_card.get_name()} at {selected_card.get_location()}")
            cards.remove(selected_card)
        return None

    elif selected_companion == 'Ramsay':
        if len(move) < 3:
            print("DEBUG: Invalid move for Ramsay, skipping.")
            return None

        first_card = find_card(cards, move[1])
        second_card = find_card(cards, move[2])

        if first_card and second_card:
            print(
                f"DEBUG: Swapping {first_card.get_name()} at {first_card.get_location()} with {second_card.get_name()} at {second_card.get_location()}")
            first_card.set_location(second_card.get_location())
            second_card.set_location(first_card.get_location())

        return None


def find_card(cards, location):
    """یافتن کارت در موقعیت مشخص"""
    for card in cards:
        if card.get_location() == location:
            return card
    return None  # اگر کارتی در آن موقعیت پیدا نشد، None برگردان

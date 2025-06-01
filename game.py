import random
from constants import DECK, RANKS, ANTE, PREFLOP_BET_SIZE, POSTFLOP_BET_SIZE
from utils import (
    generate_unique_deals, is_round_terminal, determine_showdown_winner,
    get_info_set_key, get_legal_actions, MAX_RAISES_PREFLOP, MAX_RAISES_POSTFLOP
)

def play_one_hand(adaptive_bot, opponent_fixed_policy, adaptive_update=True):
    deck_copy = list(DECK)
    p0_private_card = random.choice(deck_copy)
    deck_copy.remove(p0_private_card)
    p1_private_card = random.choice(deck_copy)
    deck_copy.remove(p1_private_card)

    public_card = None
    preflop_history_str = ""
    postflop_history_str = ""
    pot = ANTE * 2
    # random who starts first
    if random.choice([True, False]): 
        is_adaptive_bot_turn = True
    else:
        is_adaptive_bot_turn = False

    while True:
        current_player_private_card = p0_private_card if is_adaptive_bot_turn else p1_private_card
        acting_player_idx = 0 if is_adaptive_bot_turn else 1
        info_set_key = get_info_set_key(
            acting_player_idx, current_player_private_card, public_card,
            preflop_history_str, postflop_history_str
        )

        # check if game has ended
        if public_card is not None and is_round_terminal(postflop_history_str):
            winner_idx = determine_showdown_winner(p0_private_card, p1_private_card, public_card)
            # print("Game history:", preflop_history_str, postflop_history_str, "Public card:", public_card, "Bot private card:", p0_private_card, "Opponent private card:", p1_private_card)
            # print("Winner index:", winner_idx)
            if winner_idx == 0:
                # tie
                return 0
            elif winner_idx == 1:
                return pot
            else:
                return -pot
            # return pot if winner_idx == 0 else -pot
        
        if preflop_history_str.endswith('f'):
            # print("Game history:", preflop_history_str, postflop_history_str)
            return -pot if is_adaptive_bot_turn else pot
        
        if postflop_history_str and postflop_history_str.endswith('f'):
            # print("Game history:", preflop_history_str, postflop_history_str)
            return -pot if is_adaptive_bot_turn else pot

        # Game has not ended, take an action
        if acting_player_idx == 0:
            action = adaptive_bot.choose_action(info_set_key)
        else:
            opponent_strategy_at_infoset = opponent_fixed_policy.get(info_set_key)
            actions = list(opponent_strategy_at_infoset.keys())
            probabilities = list(opponent_strategy_at_infoset.values())
            action = random.choices(actions, weights=probabilities, k=1)[0]
            if adaptive_update:
                adaptive_bot.observe_opponent_action(info_set_key, action)
        bet_size_this_round = POSTFLOP_BET_SIZE if public_card else PREFLOP_BET_SIZE
        current_round_hist_for_prev_action = postflop_history_str if public_card else preflop_history_str
        previous_action_in_round = current_round_hist_for_prev_action[-1] if current_round_hist_for_prev_action else ''
        if action == 'r':
            pot += bet_size_this_round
        elif action == 'c' and previous_action_in_round == 'r':
            pot += bet_size_this_round
        if public_card is None:
            preflop_history_str += action
        else:
            postflop_history_str += action

        if public_card is None and is_round_terminal(preflop_history_str) and not preflop_history_str.endswith('f'):
            public_card = random.choice(deck_copy)

        # Action has been taken, swap turns
        is_adaptive_bot_turn = not is_adaptive_bot_turn
        acting_player_idx = 0 if is_adaptive_bot_turn else 1





    while True:
        if public_card is None and \
        is_round_terminal(preflop_history_str) and \
           not preflop_history_str.endswith('f'):
            public_card = random.choice(deck_copy)
            is_adaptive_bot_turn = False
            continue

        if preflop_history_str.endswith('f'):
            print("Game history:", preflop_history_str, postflop_history_str)
            return pot if is_adaptive_bot_turn else -pot

        if postflop_history_str and postflop_history_str.endswith('f'):
            print("Game history:", preflop_history_str, postflop_history_str)
            return pot if is_adaptive_bot_turn else -pot

        if public_card is not None and is_round_terminal(postflop_history_str):
            winner_idx = determine_showdown_winner(p0_private_card, p1_private_card, public_card)
            print("Game history:", preflop_history_str, postflop_history_str, "Public card:", public_card, "Bot private card:", p0_private_card, "Opponent private card:", p1_private_card)
            print("Winner index:", winner_idx)
            return pot if winner_idx == 1 else -pot

        
        current_player_private_card = p0_private_card if is_adaptive_bot_turn else p1_private_card

        info_set_key = get_info_set_key(
            acting_player_idx, current_player_private_card, public_card,
            preflop_history_str, postflop_history_str
        )

        if is_adaptive_bot_turn:
            chosen_action = adaptive_bot.choose_action(info_set_key)
        else:
            opponent_strategy_at_infoset = opponent_fixed_policy.get(info_set_key)
            actions = list(opponent_strategy_at_infoset.keys())
            probabilities = list(opponent_strategy_at_infoset.values())
            chosen_action = random.choices(actions, weights=probabilities, k=1)[0]
            if adaptive_update:
                adaptive_bot.observe_opponent_action(info_set_key, chosen_action)

        bet_size_this_round = POSTFLOP_BET_SIZE if public_card is not None else PREFLOP_BET_SIZE
        current_round_hist_for_prev_action = postflop_history_str if public_card else preflop_history_str
        previous_action_in_round = current_round_hist_for_prev_action[-1] if current_round_hist_for_prev_action else ''

        if chosen_action == 'r':
            pot += bet_size_this_round
        elif chosen_action == 'c' and previous_action_in_round == 'r':
            pot += bet_size_this_round

        if public_card is None:
            preflop_history_str += chosen_action
        else:
            postflop_history_str += chosen_action

        is_adaptive_bot_turn = not is_adaptive_bot_turn
        acting_player_idx = 0 if is_adaptive_bot_turn else 1

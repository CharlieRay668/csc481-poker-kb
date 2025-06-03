import random
from constants import DECK, RANKS, ANTE, PREFLOP_BET_SIZE, POSTFLOP_BET_SIZE
from utils import (
    generate_unique_deals, is_round_terminal, determine_showdown_winner,
    get_info_set_key, get_legal_actions, MAX_RAISES_PREFLOP, MAX_RAISES_POSTFLOP,
    ALL_INFOSETS
)

def play_one_hand(adaptive_bot, opponent_fixed_policy, adaptive_update=True):
    deck_copy = list(DECK)
    p0_private_card = random.choice(deck_copy)
    deck_copy.remove(p0_private_card)
    p1_private_card = random.choice(deck_copy)
    deck_copy.remove(p1_private_card)


    public_card = None
    estimated_policy = None
    preflop_history_str = ""
    postflop_history_str = ""
    pot = ANTE * 2
    # random who starts first
    if random.choice([True, False]): 
        is_adaptive_bot_turn = True
    else:
        is_adaptive_bot_turn = False

    # is_adaptive_bot_turn = True  # Adaptive bot always starts first
    # print("p0_private_card:", p0_private_card, "p1_private_card:", p1_private_card)
    while True:
        public_card_str = str(public_card) if public_card else "None"
        current_history = preflop_history_str + "/" + postflop_history_str + "/" + str(p0_private_card) + "/" + str(p1_private_card) + "/" + public_card_str
        current_player_private_card = p0_private_card if is_adaptive_bot_turn else p1_private_card
        acting_player_idx = 0 if is_adaptive_bot_turn else 1
        info_set_key = get_info_set_key(current_player_private_card, public_card,
            preflop_history_str, postflop_history_str
        )

        if preflop_history_str.endswith('f'):
            if is_adaptive_bot_turn:
                return pot, current_history, estimated_policy
            else:
                return -pot, current_history, estimated_policy
            # return pot, current_history if is_adaptive_bot_turn else -pot, current_history
        
        if postflop_history_str and postflop_history_str.endswith('f'):
            if is_adaptive_bot_turn:
                return pot, current_history, estimated_policy
            else:
                return -pot, current_history, estimated_policy
            # return pot, current_history if is_adaptive_bot_turn else -pot, current_history

        # check if game has ended
        if public_card is not None and is_round_terminal(postflop_history_str):
            winner_idx = determine_showdown_winner(p0_private_card, p1_private_card, public_card)
            if winner_idx == 0:
                # tie
                return 0, current_history, estimated_policy
            elif winner_idx == 1:
                return pot, current_history, estimated_policy
            else:
                return -pot, current_history, estimated_policy
        
        # Game has not ended, take an action
        if acting_player_idx == 0:
            action = adaptive_bot.choose_action(info_set_key)
            # print(f"Adaptive bot action at {info_set_key}: {action}")
        else:
            opponent_strategy_at_infoset = opponent_fixed_policy.get(info_set_key)
            actions = list(opponent_strategy_at_infoset.keys())
            probabilities = list(opponent_strategy_at_infoset.values())
            # print(opponent_strategy_at_infoset)
            action = random.choices(actions, weights=probabilities, k=1)[0]
            # print(f"Opponent action at {info_set_key}: {action}")
            if adaptive_update:
                adaptive_bot.observe_opponent_action(info_set_key, action)
                estimated_policy = adaptive_bot.get_opponent_belief()
        bet_size_this_round = POSTFLOP_BET_SIZE if public_card else PREFLOP_BET_SIZE
        current_round_hist_for_prev_action = postflop_history_str if public_card else preflop_history_str
        previous_action_in_round = current_round_hist_for_prev_action[-1] if current_round_hist_for_prev_action else ''
        if action == 'r':
            pot += bet_size_this_round
        elif action == 'c' and previous_action_in_round == 'r':
            pot += bet_size_this_round
        if public_card is None:
            # print("Adding preflop action:", action)
            preflop_history_str += action
        else:
            postflop_history_str += action

        if public_card is None and is_round_terminal(preflop_history_str) and not preflop_history_str.endswith('f'):
            public_card = random.choice(deck_copy)
            # print("Dealing public card:", public_card)

        # Action has been taken, swap turns
        is_adaptive_bot_turn = not is_adaptive_bot_turn
        acting_player_idx = 0 if is_adaptive_bot_turn else 1
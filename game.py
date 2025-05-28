# This file contains the logic for playing a single hand of Leduc Poker between
# an adaptive bot and an opponent with a fixed policy.

import random
from constants import (
    DECK, RANKS, ANTE, PREFLOP_BET_SIZE, POSTFLOP_BET_SIZE
)
from utils import (
    generate_unique_deals, is_round_terminal, determine_showdown_winner,
    get_info_set_key, get_legal_actions, MAX_RAISES_PREFLOP, MAX_RAISES_POSTFLOP
)

# Plays one hand of Leduc Poker.
# The 'adaptive_bot' is an instance of AdaptivePokerBot.
# 'opponent_fixed_policy' is a dictionary representing the opponent's strategy.
# Returns the payoff for the adaptive_bot (player 0).
def play_one_hand(adaptive_bot, opponent_fixed_policy):
    # 1. Deal private cards
    # Generate all unique deals once if not already done, or use a pre-generated list
    # For simplicity, random.choice from DECK ensures correct sampling if cards are replaced conceptually.
    # The original used `random.choice(unique_deals())`.
    
    # Correct dealing: pick two different cards from the deck
    deck_copy = list(DECK)
    p0_private_card = random.choice(deck_copy)
    deck_copy.remove(p0_private_card)
    p1_private_card = random.choice(deck_copy)
    
    # Initialize game state variables
    public_card_rank = None
    preflop_history_str = ""
    postflop_history_str = ""

    # Pot starts with antes from both players
    current_pot_size = ANTE * 2
    # Track total chips committed by each player to the pot during the hand
    p0_chips_committed_total = ANTE
    p1_chips_committed_total = ANTE

    # Player 0 (adaptive_bot) starts the pre-flop betting round.
    is_adaptive_bot_turn = False # Will be flipped to True for the first action

    # Main game loop for betting rounds
    while True:
        # 2. Chance Node: Reveal public card (if pre-flop ended without a fold)
        if public_card_rank is None and \
           is_round_terminal(preflop_history_str) and \
           not preflop_history_str.endswith('f'):
            
            # Randomly select a public card rank from J, Q, K.
            # The original was `random.choice([x for x in 'JQK' if x not in (c0,c1)])`
            # This implies public card cannot be same rank as any private card.
            # Standard Leduc: public card is one of J, Q, K. Deck has 2 of each.
            # If one J is private, another J can be public.
            # Let's stick to "any rank can be public card" for simplicity and common interpretation.
            public_card_rank = random.choice(RANKS)
            
            # Player 0 starts the post-flop betting round.
            is_adaptive_bot_turn = False # Will be flipped to True
            continue # Restart loop for post-flop round

        # 3. Check for Terminal States (Fold or Showdown)
        # Fold in pre-flop round
        if preflop_history_str.endswith('f'):
            # If it was P1's turn (adaptive_bot just acted and P1 folded), bot wins P1's committed chips.
            # If it was adaptive_bot's turn (P1 acted and bot folded), bot loses its committed chips.
            # `is_adaptive_bot_turn` here refers to whose turn it *would have been*.
            # If bot folded, `is_adaptive_bot_turn` would be True (it was its turn, it folded).
            # Payoff is from bot's perspective.
            return p1_chips_committed_total if is_adaptive_bot_turn else -p0_chips_committed_total

        # Fold in post-flop round
        if postflop_history_str and postflop_history_str.endswith('f'):
            return p1_chips_committed_total if is_adaptive_bot_turn else -p0_chips_committed_total

        # Showdown (post-flop round ended without a fold)
        if public_card_rank is not None and is_round_terminal(postflop_history_str):
            winner_idx = determine_showdown_winner(p0_private_card, p1_private_card, public_card_rank)
            # Payoff: if bot wins (winner_idx == 1), it gets P1's committed chips.
            # If bot loses (winner_idx == -1), it loses its own committed chips.
            return p1_chips_committed_total if winner_idx == 1 else -p0_chips_committed_total

        # 4. Determine Current Player and Get Action
        is_adaptive_bot_turn = not is_adaptive_bot_turn # Switch player
        
        acting_player_idx = 0 if is_adaptive_bot_turn else 1
        current_player_private_card = p0_private_card if is_adaptive_bot_turn else p1_private_card
        
        # Construct the info set key for the current player
        info_set_key = get_info_set_key(
            acting_player_idx, current_player_private_card, public_card_rank,
            preflop_history_str, postflop_history_str
        )

        # Get action from the appropriate player
        chosen_action = None
        if is_adaptive_bot_turn: # Adaptive bot's turn
            chosen_action = adaptive_bot.choose_action(info_set_key)
        else: # Opponent's turn
            opponent_strategy_at_infoset = opponent_fixed_policy.get(info_set_key)
            if not opponent_strategy_at_infoset: # Should be covered by complete_policy
                # Fallback if somehow opponent policy is incomplete for this infoset
                current_round_hist = postflop_history_str if public_card_rank else preflop_history_str
                max_r = MAX_RAISES_POSTFLOP if public_card_rank else MAX_RAISES_PREFLOP
                legal_acts = get_legal_actions(current_round_hist, max_r)
                chosen_action = random.choice(legal_acts) if legal_acts else None # Should not be None here
            else:
                actions = list(opponent_strategy_at_infoset.keys())
                probabilities = list(opponent_strategy_at_infoset.values())
                if not actions: # Should not happen
                    current_round_hist = postflop_history_str if public_card_rank else preflop_history_str
                    max_r = MAX_RAISES_POSTFLOP if public_card_rank else MAX_RAISES_PREFLOP
                    legal_acts = get_legal_actions(current_round_hist, max_r)
                    chosen_action = random.choice(legal_acts) if legal_acts else 'f' # emergency fold
                else:
                    chosen_action = random.choices(actions, weights=probabilities, k=1)[0]
            
            # Adaptive bot observes the opponent's action
            adaptive_bot.observe_opponent_action(info_set_key, chosen_action)

        if chosen_action is None: # Should only happen if choose_action itself fails or terminal state was missed
            # This indicates an issue, potentially trying to act in a terminal state not caught above.
            # Or a bug in policy providing no actions.
            # For robustness, assume fold if no action decided.
            chosen_action = 'f' 


        # 5. Update Game State based on Action
        bet_size_this_round = POSTFLOP_BET_SIZE if public_card_rank is not None else PREFLOP_BET_SIZE
        
        # Get previous action in the current round for 'call' logic
        current_round_hist_for_prev_action = postflop_history_str if public_card_rank else preflop_history_str
        previous_action_in_round = current_round_hist_for_prev_action[-1] if current_round_hist_for_prev_action else ''

        if chosen_action == 'r': # Raise
            current_pot_size += bet_size_this_round
            if is_adaptive_bot_turn:
                p0_chips_committed_total += bet_size_this_round
            else:
                p1_chips_committed_total += bet_size_this_round
        elif chosen_action == 'c': # Call or Check
            # If previous action was a raise, this 'c' is a call, so chips are added.
            if previous_action_in_round == 'r':
                current_pot_size += bet_size_this_round
                if is_adaptive_bot_turn:
                    p0_chips_committed_total += bet_size_this_round
                else:
                    p1_chips_committed_total += bet_size_this_round
            # If previous action was not a raise, this 'c' is a check, no chips added by this player now.

        # Append action to the appropriate history string
        if public_card_rank is None: # Pre-flop round
            preflop_history_str += chosen_action
        else: # Post-flop round
            postflop_history_str += chosen_action
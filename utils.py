# This file contains utility functions for game logic, information set generation,
# and policy manipulation used across different parts of the Leduc Poker bot.

import random
from constants import RANKS, DECK, RANK_VALUE, MAX_RAISES_PREFLOP, MAX_RAISES_POSTFLOP

# Generates all unique two-card deals (player private cards) from the deck.
# Each deal is a tuple (player0_card, player1_card).
def generate_unique_deals():
    deals = []
    # Iterate through all possible card combinations for player 0
    for i, card0 in enumerate(DECK):
        # Iterate through all possible card combinations for player 1
        for j, card1 in enumerate(DECK):
            # Ensure the cards are different (drawn from distinct positions in the deck)
            if i != j:
                deals.append((card0, card1))
    return deals

# Determines if a game state, represented by its history string for a single round, is terminal.
# A round's history is terminal if a player folds ('f'),
# or if betting concludes with checks ('cc') or a call after a raise ('rc').
def is_round_terminal(round_history_string):
    # Check for a fold action, which immediately ends the betting in the round
    if round_history_string.endswith('f'):
        return True
    # Check for sequences indicating the end of betting:
    # 'cc': both players check (or player 2 checks after player 1 checks)
    # 'rc': player 2 calls player 1's raise (or player 1 calls player 2's raise)
    if len(round_history_string) >= 2 and round_history_string[-2:] in ('cc', 'rc'):
        return True
    return False

# Returns a list of legal actions for the current player given the round's history and raise limit.
def get_legal_actions(round_history_string, max_raises_for_round):
    # If the round is already terminal, no actions are legal.
    if is_round_terminal(round_history_string):
        return []

    num_raises_in_round = round_history_string.count('r')

    # Determine if there's an open bet (a bet or raise that hasn't been called or folded to).
    # This occurs if the history is not empty and the last action was a raise ('r'),
    # or if a raise ('r') exists and its last occurrence is after the last call ('c').
    is_bet_open = False
    if round_history_string: # Check if history is not empty
        if round_history_string[-1] == 'r':
            is_bet_open = True
        elif 'r' in round_history_string:
            last_raise_index = round_history_string.rfind('r')
            last_call_index = round_history_string.rfind('c')
            if last_raise_index > last_call_index:
                is_bet_open = True

    # If there's no open bet, player can check or bet/raise (if not at max raises).
    if not is_bet_open:
        actions = ['c'] # Check/Call (in this context, 'c' means check)
        if num_raises_in_round < max_raises_for_round:
            actions.append('r') # Bet/Raise
        return actions
    # If there is an open bet, player can call, fold, or raise (if not at max raises).
    else:
        actions = ['c', 'f'] # Call, Fold
        if num_raises_in_round < max_raises_for_round:
            actions.append('r') # Raise
        return actions

# Determines the winner at showdown. Returns +1 if player 0 wins, -1 if player 1 wins.
# Assumes player 0 has p0_card and player 1 has p1_card.
def determine_showdown_winner(p0_card, p1_card, public_card):
    p0_has_pair = (p0_card == public_card)
    p1_has_pair = (p1_card == public_card)

    # A pair with the board card is the strongest hand.
    # If player 0 has a pair and player 1 does not, player 0 wins.
    if p0_has_pair and not p1_has_pair:
        return 1
    # If player 1 has a pair and player 0 does not, player 1 wins.
    if p1_has_pair and not p0_has_pair:
        return -1
    # if neither player has a pair, the winner is determined by the rank of their private card.
    # (In the case of both having a pair, they both pair the same public card, so their private cards break the tie)
    # if there is a tie, return 0
    if RANK_VALUE[p0_card] == RANK_VALUE[p1_card]:
        return 0
    if RANK_VALUE[p0_card] > RANK_VALUE[p1_card]:
        return 1
    else: # RANK_VALUE[p1_card] > RANK_VALUE[p0_card] (cards are unique, so values can't be equal)
        return -1


# Generates a unique key for an information set.
# An information set groups game states where a player has the same information.
def get_info_set_key(player_index, private_card_rank, public_card_rank, preflop_history, postflop_history):
    # Pre-flop round: public card is None. History is just preflop_history.
    if public_card_rank is None:
        return (player_index, private_card_rank, None, preflop_history)
    # Post-flop round: public card is known. History includes preflop and postflop actions, separated by '/'.
    else:
        return (player_index, private_card_rank, public_card_rank, f'{preflop_history}/{postflop_history}')

# Enumerates all possible information sets in the game.
def enumerate_all_infosets():
    # Helper function to recursively generate all possible round histories.
    def _generate_round_histories(max_raises_for_round):
        histories = set()
        # Recursive helper to build histories
        def _recursive_build(current_history, num_raises, is_bet_currently_open):
            histories.add(current_history)
            # Stop if the current history is terminal for the round
            if is_round_terminal(current_history):
                return
            # Determine legal moves based on current state
            possible_actions = get_legal_actions(current_history, max_raises_for_round) # Use the outer scope's get_legal_actions

            for action in possible_actions:
                _recursive_build(
                    current_history + action,
                    num_raises + (1 if action == 'r' else 0),
                    action == 'r' # A raise action opens the bet
                )
        _recursive_build('', 0, False) # Start with empty history, 0 raises, no open bet
        return histories

    # Generate all pre-flop and post-flop round histories
    preflop_round_histories = _generate_round_histories(MAX_RAISES_PREFLOP)
    postflop_round_histories = _generate_round_histories(MAX_RAISES_POSTFLOP)

    all_infosets = []

    # Pre-flop information sets
    for player_idx in (0, 1): # For each player
        for private_card_r in RANKS: # For each possible private card rank
            # Iterate through non-terminal pre-flop histories
            for pre_hist in [h for h in preflop_round_histories if not is_round_terminal(h)]:
                all_infosets.append(get_info_set_key(player_idx, private_card_r, None, pre_hist, ''))

    # Post-flop information sets
    # These occur after a pre-flop round that didn't end in a fold.
    preflop_histories_leading_to_postflop = [
        h for h in preflop_round_histories if is_round_terminal(h) and not h.endswith('f')
    ]
    for player_idx in (0, 1): # For each player
        for private_card_r in RANKS: # For each possible private card rank
            for pre_hist in preflop_histories_leading_to_postflop: # For each pre-flop history that leads to post-flop
                for public_card_r in RANKS: # For each possible public card rank
                    # Iterate through non-terminal post-flop histories
                    for post_hist in [h for h in postflop_round_histories if not is_round_terminal(h)]:
                        all_infosets.append(get_info_set_key(player_idx, private_card_r, public_card_r, pre_hist, post_hist))
    return all_infosets

# Pre-calculate all information sets and legal actions for each.
# This is done once and imported by other modules to avoid re-computation.
ALL_INFOSETS = enumerate_all_infosets()

LEGAL_ACTIONS_AT_INFOSET = {
    info_set_key: get_legal_actions(
        info_set_key[3].split('/')[-1], # Current round's history string (part after '/' if post-flop)
        MAX_RAISES_POSTFLOP if info_set_key[2] is not None else MAX_RAISES_PREFLOP # Max raises depends on round
    )
    for info_set_key in ALL_INFOSETS
}

# Takes a potentially incomplete policy dictionary and ensures it covers all info sets.
# If an info set is missing from the input policy, it's assigned a uniform random strategy.
def complete_policy(partial_policy_dict):
    full_policy = dict(partial_policy_dict) # Create a shallow copy to modify
    # Iterate through all theoretically possible information sets
    for info_set_key in ALL_INFOSETS:
        # If the current info set is not covered by the input policy
        if info_set_key not in full_policy:
            # Get the legal actions for this info set
            legal_actions_list = LEGAL_ACTIONS_AT_INFOSET[info_set_key]
            # Assign a uniform random strategy: each legal action has equal probability
            if legal_actions_list: # Ensure there are legal actions
                full_policy[info_set_key] = {action: 1.0 / len(legal_actions_list) for action in legal_actions_list}
            else: # Should not happen for non-terminal infosets generated by enumerate_all_infosets
                full_policy[info_set_key] = {}
    return full_policy
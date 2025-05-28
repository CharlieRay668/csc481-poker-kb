# Implements the Counterfactual Regret Minimization (CFR) algorithm.
# This version calculates utilities based on chip differences and can be trained incrementally.

import random
from collections import defaultdict
from utils import (
    is_round_terminal, determine_showdown_winner, get_info_set_key,
    LEGAL_ACTIONS_AT_INFOSET, generate_unique_deals
)
from constants import PREFLOP_BET_SIZE, POSTFLOP_BET_SIZE, RANKS


class CFRTrainer:
    def __init__(self):
        # Stores the cumulative regrets for not choosing each action at each information set.
        # Structure: {info_set_key: {action: cumulative_regret_value}}
        self.cumulative_regrets = defaultdict(lambda: defaultdict(float))

        # Stores the sum of strategy probabilities chosen at each information set over all iterations.
        # Used to compute the average strategy, which converges to Nash Equilibrium.
        # Structure: {info_set_key: {action: sum_of_probabilities}}
        self.strategy_sum_across_iterations = defaultdict(lambda: defaultdict(float))

    # Calculates the current strategy for an information set based on positive regrets.
    # If all regrets are non-positive, it returns a uniform random strategy.
    def _calculate_current_strategy_profile(self, info_set_key):
        legal_actions_list = LEGAL_ACTIONS_AT_INFOSET[info_set_key]
        current_strategy = {}

        # Get regrets for legal actions, ensuring non-negativity.
        positive_regrets = [max(self.cumulative_regrets[info_set_key][action], 0) for action in legal_actions_list]
        sum_positive_regrets = sum(positive_regrets)

        # If sum of positive regrets is zero, use a uniform random strategy.
        if sum_positive_regrets == 0:
            num_actions = len(legal_actions_list)
            for action in legal_actions_list:
                current_strategy[action] = 1.0 / num_actions
        # Otherwise, normalize positive regrets to form the strategy.
        else:
            for i, action in enumerate(legal_actions_list):
                current_strategy[action] = positive_regrets[i] / sum_positive_regrets
        return current_strategy

    # Helper function to calculate chip movements for an action during traversal.
    # Returns: (pot_increase, p0_chips_this_action, p1_chips_this_action)
    def _get_chip_payment_for_action(self, action, previous_action_in_round, is_player0_acting, public_card_rank):
        bet_size = PREFLOP_BET_SIZE if public_card_rank is None else POSTFLOP_BET_SIZE
        
        pot_increase_val = 0
        p0_chips_action_val = 0
        p1_chips_action_val = 0

        if action == 'r':  # Raise action
            pot_increase_val = bet_size
            if is_player0_acting:
                p0_chips_action_val = bet_size
            else:
                p1_chips_action_val = bet_size
        elif action == 'c' and previous_action_in_round == 'r':  # Call a raise
            pot_increase_val = bet_size
            if is_player0_acting:
                p0_chips_action_val = bet_size
            else:
                p1_chips_action_val = bet_size
        # 'c' (check) or 'f' (fold) don't involve immediate payment from this player in this model
        return pot_increase_val, p0_chips_action_val, p1_chips_action_val


    # Performs a single traversal of the game tree for CFR.
    # p0_card, p1_card: private cards for player 0 and player 1.
    # public_card_rank: rank of the public card (None if pre-flop).
    # preflop_hist_str, postflop_hist_str: history strings for pre-flop and post-flop rounds.
    # current_pot: current size of the pot.
    # p0_chips_committed, p1_chips_committed: chips committed by each player so far *in this hand*.
    # acting_player_idx: index of the player whose turn it is (0 or 1).
    # fixed_opponent_policy: if not None, player 1 (opponent) uses this fixed policy.
    #                        if None, both players use the CFR learner's current strategy (self-play).
    # Returns the expected utility (value) of the current game state for player 0.
    def traverse_game_tree(self, p0_card, p1_card, public_card_rank,
                           preflop_hist_str, postflop_hist_str,
                           current_pot, p0_chips_committed, p1_chips_committed,
                           acting_player_idx, fixed_opponent_policy):

        # 1. Handle Terminal States
        # If pre-flop history ends in a fold
        if preflop_hist_str.endswith('f'):
            # If player 0 (learner, if acting_player_idx was 0 before fold) folded, utility is -p0_chips_committed.
            # If player 1 folded (acting_player_idx was 1), utility for P0 is pot - p0_chips_committed.
            # The 'acting_player_idx' is the player *who would have acted next*.
            # So if it's P0's turn to act and P1 just folded, P0 wins.
            # If P0 just folded, it's P1's "turn" conceptually, P0 loses.
            is_p0_turn_to_act_if_not_fold = (acting_player_idx == 0)
            return current_pot - p0_chips_committed if not is_p0_turn_to_act_if_not_fold else -p0_chips_committed

        # If post-flop history ends in a fold
        if postflop_hist_str and postflop_hist_str.endswith('f'):
            is_p0_turn_to_act_if_not_fold = (acting_player_idx == 0)
            return current_pot - p0_chips_committed if not is_p0_turn_to_act_if_not_fold else -p0_chips_committed
        
        # If post-flop round is terminal (showdown or call/check sequence)
        if public_card_rank is not None and is_round_terminal(postflop_hist_str):
            winner = determine_showdown_winner(p0_card, p1_card, public_card_rank)
            return (current_pot - p0_chips_committed) if winner == 1 else -p0_chips_committed

        # 2. Handle Chance Node: Revealing the public card (flop)
        # This occurs if pre-flop is terminal (but not a fold) and public card hasn't been revealed yet.
        if public_card_rank is None and is_round_terminal(preflop_hist_str) and not preflop_hist_str.endswith('f'):
            expected_utility_from_chance_node = 0
            # Possible public cards are those not dealt as private cards
            possible_public_cards = [r for r in RANKS if r != p0_card and r != p1_card]
            # If somehow no public cards are possible (e.g., J,J dealt and looking for J pub), this needs care.
            # Leduc DECK has 2 of each, so if p0, p1 are same rank, that rank can still be public.
            # The original used `ranks = {'J', 'Q', 'K'} - {c0, c1}`. This is for distinct ranks.
            # For Leduc, any of the 3 ranks can be a public card. Assume uniform.
            
            # Correct logic for public card: any of the 3 ranks.
            # The specific cards remaining in deck don't affect probability of J, Q, or K rank appearing, assuming one of each is available.
            # The problem statement used "uniform over ranks", which is simpler.
            num_possible_public_ranks = len(RANKS)
            for pub_r in RANKS:
                expected_utility_from_chance_node += self.traverse_game_tree(
                    p0_card, p1_card, pub_r, preflop_hist_str, '', # Start empty postflop history
                    current_pot, p0_chips_committed, p1_chips_committed,
                    0, fixed_opponent_policy # Player 0 starts post-flop betting
                )
            return expected_utility_from_chance_node / num_possible_public_ranks

        # 3. Handle Decision Node
        current_player_card = p0_card if acting_player_idx == 0 else p1_card
        info_set_key = get_info_set_key(acting_player_idx, current_player_card, public_card_rank,
                                        preflop_hist_str, postflop_hist_str)
        
        legal_actions_list = LEGAL_ACTIONS_AT_INFOSET[info_set_key]
        if not legal_actions_list: # Should not happen in a non-terminal state
             return 0 # Or raise error

        current_round_history_str = postflop_hist_str if public_card_rank is not None else preflop_hist_str

        # Get strategy for the current player at this info set
        player_strategy_profile = self._calculate_current_strategy_profile(info_set_key)

        # If player 1 (opponent) is acting and a fixed policy is provided, use it.
        if acting_player_idx == 1 and fixed_opponent_policy is not None:
            # Use opponent's fixed strategy, fallback to uniform if info set missing (should be completed)
            strategy_to_use = fixed_opponent_policy.get(info_set_key, 
                                                     {a: 1/len(legal_actions_list) for a in legal_actions_list})
        else: # Player 0 (learner) is acting, or it's self-play (fixed_opponent_policy is None)
            strategy_to_use = player_strategy_profile
        
        # The value of the current node (state) is the sum of (action_prob * action_value)
        node_expected_value = 0
        # Store counterfactual values for each action to update regrets later
        action_counterfactual_values = {action: 0 for action in legal_actions_list}

        for action in legal_actions_list:
            # Store current pot and chip commitments to revert for next counterfactual calculation
            pot_before_this_action = current_pot
            p0_chips_before_this_action = p0_chips_committed
            p1_chips_before_this_action = p1_chips_committed

            # Determine chip movements for this action
            previous_action_in_round = current_round_history_str[-1] if current_round_history_str else ''
            
            pot_increase, p0_action_chips, p1_action_chips = self._get_chip_payment_for_action(
                action, previous_action_in_round, acting_player_idx == 0, public_card_rank
            )

            # Update pot and chip commitments based on the action
            new_pot = pot_before_this_action + pot_increase
            new_p0_chips_committed = p0_chips_before_this_action + p0_action_chips
            new_p1_chips_committed = p1_chips_before_this_action + p1_action_chips
            
            # Update history strings
            next_preflop_hist_str = preflop_hist_str + action if public_card_rank is None else preflop_hist_str
            next_postflop_hist_str = postflop_hist_str + action if public_card_rank is not None else postflop_hist_str

            # Recursively call traverse for the next state
            # Utility is from P0's perspective. If P1 (acting_player_idx=1) made the move, the returned value is already P0's utility.
            # If P0 (acting_player_idx=0) made the move, the returned value is also P0's utility.
            value_after_action = self.traverse_game_tree(
                p0_card, p1_card, public_card_rank,
                next_preflop_hist_str, next_postflop_hist_str,
                new_pot, new_p0_chips_committed, new_p1_chips_committed,
                1 - acting_player_idx, # Switch player turn
                fixed_opponent_policy
            )
            
            action_counterfactual_values[action] = value_after_action
            # Accumulate node's expected value, weighted by probability of taking this action
            node_expected_value += strategy_to_use[action] * value_after_action
        
        # 4. Update Regrets and Strategy Sum (if learner's turn or self-play)
        # Regrets are updated for the player whose decision node this is.
        # If fixed_opponent_policy is None (self-play), update for both players.
        # If fixed_opponent_policy is not None, only update for player 0 (the learner).
        if fixed_opponent_policy is None or acting_player_idx == 0:
            for action in legal_actions_list:
                # Regret is counterfactual_value_of_action - node_expected_value.
                # For P0, this is direct.
                # For P1 (acting_player_idx=1) in self-play, utilities are from P0's perspective.
                # So, a higher utility for P0 means a lower utility for P1.
                # Regret for P1 = (P1's actual value for action) - (P1's expected value for node)
                #              = (-P0's value for action) - (-P0's expected value for node)
                #              = node_expected_value - action_counterfactual_values[action]
                regret_for_action = action_counterfactual_values[action] - node_expected_value
                if acting_player_idx == 1: # If it's P1's turn, flip sign for P1's regret
                    regret_for_action = -regret_for_action

                self.cumulative_regrets[info_set_key][action] += regret_for_action
                # Update sum of strategy probabilities (used for average strategy)
                self.strategy_sum_across_iterations[info_set_key][action] += player_strategy_profile[action]
        
        return node_expected_value


    # Trains the CFR model for a specified number of iterations against a fixed opponent policy.
    # If fixed_opponent_policy is None, it performs self-play.
    def train_iterations(self, num_iterations, fixed_opponent_policy=None):
        unique_card_deals = generate_unique_deals()
        for _ in range(num_iterations):
            # Randomly select a deal (private cards for P0 and P1)
            p0_card, p1_card = random.choice(unique_card_deals)
            
            # Initial game state: pot=2 (antes), each player committed 1.
            # Player 0 starts the pre-flop betting round.
            # The original `traverse` was called with (c0, c1, None, '', '', 2, 1, 1, 0, opp_pol)
            # The pot (2) is the total amount. p0_chips_committed (1) and p1_chips_committed (1) are correct.

            if fixed_opponent_policy is None: # Self-play mode
                # In self-play, we need to traverse from both players' perspectives to update their regrets correctly.
                # However, the original `_dual_traverse` called `traverse` twice with `acting_player_idx = 0`
                # and then `acting_player_idx = 1` from the start.
                # The current `traverse_game_tree` inherently handles regret updates for the `acting_player_idx`
                # when `fixed_opponent_policy` is `None`. So, one call from P0's start is sufficient.
                # Let's re-check original _dual_traverse:
                # self.traverse(c0, c1, None, '', '', 2, 1, 1, 0, None) -> updates P0 & P1 regs
                # self.traverse(c0, c1, None, '', '', 2, 1, 1, 1, None) -> updates P1 & P0 regs, from P1 start.
                # This seems redundant if traverse correctly assigns regrets.
                # The key is that `traverse` returns utility for P0.
                # When `acting_player_idx` is P1, `regret_for_action` is already flipped for P1.
                # The original `_dual_traverse` used in `selfplay` was:
                # self.traverse(c0, c1, None, '', '', 2, 1, 1, 0, None)
                # self.traverse(c0, c1, None, '', '', 2, 1, 1, 1, None) <-- This starts game as if P1 is first to act.
                # This is incorrect for Leduc where P0 always starts preflop.
                # The purpose of `_dual_traverse` was likely to ensure symmetric updates in some CFR variants.
                # For standard CFR, one traversal from P0's start, with `fixed_opponent_policy=None`,
                # correctly updates regrets for *both* players as they take their turns.
                # The original code's `update = (opp_pol is None) or (pl == 0)` in traverse
                # meant: if self-play, always update. if training vs fixed, only update for learner (pl=0).
                # My logic for `if fixed_opponent_policy is None or acting_player_idx == 0:` covers this.
                # So, for self-play, one call is enough.

                self.traverse_game_tree(p0_card, p1_card, None, '', '', 2, 1, 1, 0, None)

            else: # Training against a fixed opponent
                # Player 0 is the learner, Player 1 uses fixed_opponent_policy.
                # Initial pot = 2 (antes), p0_chips_committed = 1, p1_chips_committed = 1. Player 0 starts.
                self.traverse_game_tree(p0_card, p1_card, None, '', '', 2, 1, 1, 0, fixed_opponent_policy)


    # Computes the average strategy from the sum of strategy probabilities accumulated during training.
    # This average strategy is what converges to a Nash Equilibrium.
    def calculate_average_strategy(self):
        average_strategy_profile = {}
        for info_set_key, action_sums in self.strategy_sum_across_iterations.items():
            total_sum_for_infoset = sum(action_sums.values())
            if total_sum_for_infoset > 0:
                average_strategy_profile[info_set_key] = {
                    action: prob_sum / total_sum_for_infoset
                    for action, prob_sum in action_sums.items()
                }
            else:
                # Fallback for infosets that might not have been visited or had strategy sums.
                # This should ideally not happen if training is sufficient and covers states.
                legal_actions_list = LEGAL_ACTIONS_AT_INFOSET.get(info_set_key, [])
                if legal_actions_list:
                    average_strategy_profile[info_set_key] = {
                        action: 1.0 / len(legal_actions_list) for action in legal_actions_list
                    }
                else:
                    average_strategy_profile[info_set_key] = {} # Should be terminal
        return average_strategy_profile

    # Runs self-play for a number of iterations to compute an approximate Nash Equilibrium strategy.
    def run_self_play(self, iterations=60000):
        # In self-play mode, fixed_opponent_policy is None.
        # The original selfplay logic had a _dual_traverse which started a game once with P0 to act
        # and once with P1 to act. P1 never starts a Leduc round first.
        # Standard CFR self-play involves a single traversal per game, updating regrets for whomever's turn it is.
        # The traverse_game_tree with fixed_opponent_policy=None handles this.
        print(f"Running self-play for {iterations} iterations...")
        self.train_iterations(num_iterations=iterations, fixed_opponent_policy=None)
        print("Self-play complete. Calculating average strategy.")
        return self.calculate_average_strategy()
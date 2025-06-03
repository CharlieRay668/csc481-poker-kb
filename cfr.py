import random
from collections import defaultdict
from utils import (
    is_round_terminal, determine_showdown_winner, get_info_set_key,
    LEGAL_ACTIONS_AT_INFOSET, generate_unique_deals
)
from constants import PREFLOP_BET_SIZE, POSTFLOP_BET_SIZE, RANKS


class CFRTrainer:
    def __init__(self):
        self.cumulative_regrets = defaultdict(lambda: defaultdict(float))
        self.strategy_sum_across_iterations = defaultdict(lambda: defaultdict(float))

    def _calculate_current_strategy_profile(self, info_set_key):
        legal_actions_list = LEGAL_ACTIONS_AT_INFOSET[info_set_key]
        current_strategy = {}
        positive_regrets = [max(self.cumulative_regrets[info_set_key][action], 0) for action in legal_actions_list]
        sum_positive_regrets = sum(positive_regrets)

        if sum_positive_regrets == 0:
            num_actions = len(legal_actions_list)
            for action in legal_actions_list:
                current_strategy[action] = 1.0 / num_actions
        else:
            for i, action in enumerate(legal_actions_list):
                current_strategy[action] = positive_regrets[i] / sum_positive_regrets
        return current_strategy

    def _get_chip_payment_for_action(self, action, previous_action_in_round, is_player0_acting, public_card_rank):
        bet_size = PREFLOP_BET_SIZE if public_card_rank is None else POSTFLOP_BET_SIZE
        pot_increase_val = 0
        p0_chips_action_val = 0
        p1_chips_action_val = 0

        if action == 'r':
            pot_increase_val = bet_size
            if is_player0_acting:
                p0_chips_action_val = bet_size
            else:
                p1_chips_action_val = bet_size
        elif action == 'c' and previous_action_in_round == 'r':
            pot_increase_val = bet_size
            if is_player0_acting:
                p0_chips_action_val = bet_size
            else:
                p1_chips_action_val = bet_size
        return pot_increase_val, p0_chips_action_val, p1_chips_action_val

    def traverse_game_tree(self, p0_card, p1_card, public_card_rank,
                           preflop_hist_str, postflop_hist_str,
                           current_pot, p0_chips_committed, p1_chips_committed,
                           acting_player_idx, fixed_opponent_policy,
                           reach_p0: float, reach_p1: float): # Added reach probabilities


        if preflop_hist_str.endswith('f'):
            # If P0 just folded, next turn is P1's (acting_player_idx == 1). P0 loses current_pot.
            # If P1 just folded, next turn is P0's (acting_player_idx == 0). P0 wins current_pot.
            if acting_player_idx == 1: # P0 folded
                return -current_pot
            else: # P1 folded (acting_player_idx == 0)
                return current_pot

        # If post-flop history ends in a fold
        if postflop_hist_str and postflop_hist_str.endswith('f'):
            if acting_player_idx == 1: # P0 folded
                return -current_pot
            else: # P1 folded (acting_player_idx == 0)
                return current_pot
        
        # If post-flop round is terminal (showdown)
        # Assumes determine_showdown_winner returns: 1 for P0 win, 2 for P1 win, 0 for Tie.
        if public_card_rank is not None and is_round_terminal(postflop_hist_str):
            winner = determine_showdown_winner(p0_card, p1_card, public_card_rank)
            if winner == 1:  # P0 wins
                return current_pot
            elif winner == -1:  # P1 wins (P0 loses)
                return -current_pot
            elif winner == 0:  # Tie
                return 0.0
            else: # Should not happen with standard Leduc rules
                raise ValueError(f"Unknown winner index from determine_showdown_winner: {winner}")


        # 2. Handle Chance Node: Revealing the public card (flop)
        if public_card_rank is None and is_round_terminal(preflop_hist_str) and not preflop_hist_str.endswith('f'):
            expected_utility_from_chance_node = 0
            num_possible_public_ranks = len(RANKS) # In Leduc, any rank can appear
            
            for pub_r in RANKS:
                # Pass reach probabilities through chance nodes
                expected_utility_from_chance_node += self.traverse_game_tree(
                    p0_card, p1_card, pub_r, preflop_hist_str, '', 
                    current_pot, p0_chips_committed, p1_chips_committed,
                    0, fixed_opponent_policy, # Player 0 starts post-flop betting
                    reach_p0, reach_p1 # Reach probs don't change for opponent due to chance event
                )
            return expected_utility_from_chance_node / num_possible_public_ranks

        # 3. Handle Decision Node
        current_player_card = p0_card if acting_player_idx == 0 else p1_card
        info_set_key = get_info_set_key(current_player_card, public_card_rank,
                                        preflop_hist_str, postflop_hist_str)
        
        legal_actions_list = LEGAL_ACTIONS_AT_INFOSET[info_set_key]
        if not legal_actions_list:
             return 0.0 # Or raise error, should not happen in non-terminal valid state

        current_round_history_str = postflop_hist_str if public_card_rank is not None else preflop_hist_str
        player_strategy_profile = self._calculate_current_strategy_profile(info_set_key)

        strategy_to_use = player_strategy_profile
        if acting_player_idx == 1 and fixed_opponent_policy is not None:
            strategy_to_use = fixed_opponent_policy.get(info_set_key, 
                                                     {a: 1/len(legal_actions_list) for a in legal_actions_list})
        
        node_expected_value = 0.0
        action_counterfactual_values = {action: 0.0 for action in legal_actions_list}

        for action in legal_actions_list:
            action_prob = strategy_to_use[action]
            
            # Calculate new reach probabilities for the next state
            new_reach_p0 = reach_p0 * (action_prob if acting_player_idx == 0 else 1.0)
            new_reach_p1 = reach_p1 * (action_prob if acting_player_idx == 1 else 1.0)

            # Optimization: if a reach probability becomes zero, this path won't contribute.
            # However, for numerical stability and to avoid division by zero later if not handled,
            # it's often simpler to proceed. If new_reach_p0 * new_reach_p1 == 0, can prune,
            # but typically CFR proceeds as long as one player's reach is positive.
            # Here, we'll only prune if the *next state's effective reach* is zero for regret updates.
            # The recursion must continue if either reach_self or reach_opp for current node is > 0.

            pot_increase, p0_action_chips, p1_action_chips = self._get_chip_payment_for_action(
                action, 
                current_round_history_str[-1] if current_round_history_str else '', 
                acting_player_idx == 0, 
                public_card_rank
            )

            new_pot = current_pot + pot_increase
            new_p0_chips_committed = p0_chips_committed + p0_action_chips
            new_p1_chips_committed = p1_chips_committed + p1_action_chips
            
            next_preflop_hist_str = preflop_hist_str + action if public_card_rank is None else preflop_hist_str
            next_postflop_hist_str = postflop_hist_str + action if public_card_rank is not None else postflop_hist_str

            value_after_action = self.traverse_game_tree(
                p0_card, p1_card, public_card_rank,
                next_preflop_hist_str, next_postflop_hist_str,
                new_pot, new_p0_chips_committed, new_p1_chips_committed,
                1 - acting_player_idx,
                fixed_opponent_policy,
                new_reach_p0, new_reach_p1 # Pass updated reach probs
            )
            
            action_counterfactual_values[action] = value_after_action
            node_expected_value += action_prob * value_after_action
        
        # 4. Update Regrets and Strategy Sum
        # Update only if it's the learner's turn or self-play for the current acting_player_idx
        if fixed_opponent_policy is None or acting_player_idx == 0:
            # Determine whose reach probability to use for weighting
            # Regrets are weighted by opponent's reach to this state
            # Strategy sum is weighted by current player's reach to this state
            current_player_reach = reach_p0 if acting_player_idx == 0 else reach_p1
            opponent_reach = reach_p1 if acting_player_idx == 0 else reach_p0

            if current_player_reach > 0: # Update strategy sum only if player could reach this state
                 for action in legal_actions_list:
                    self.strategy_sum_across_iterations[info_set_key][action] += \
                        current_player_reach * player_strategy_profile[action]
            
            if opponent_reach > 0: # Update regrets only if opponent could reach this state
                for action in legal_actions_list:
                    # Regret is (value of taking action) - (expected value of the node)
                    # From the perspective of the acting player.
                    # Since all utilities are from P0's perspective:
                    regret_for_action = action_counterfactual_values[action] - node_expected_value
                    if acting_player_idx == 1: # If it's P1's turn, flip sign for P1's regret
                        regret_for_action = -regret_for_action

                    self.cumulative_regrets[info_set_key][action] += opponent_reach * regret_for_action
        
        return node_expected_value

    def train_iterations(self, num_iterations, fixed_opponent_policy=None):
        unique_card_deals = generate_unique_deals()
        for i in range(num_iterations):
            p0_card, p1_card = random.choice(unique_card_deals)
            
            # Initial call with reach_p0 = 1.0 and reach_p1 = 1.0
            self.traverse_game_tree(
                p0_card, p1_card, None, '', '', 
                2, 1, 1, # pot, p0_comm, p1_comm
                0, # P0 starts
                fixed_opponent_policy,
                1.0, 1.0 # initial reach_p0, reach_p1
            )

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
                legal_actions_list = LEGAL_ACTIONS_AT_INFOSET.get(info_set_key, [])
                if legal_actions_list: # Fallback to uniform if sum is 0 (e.g. not visited with positive reach)
                    average_strategy_profile[info_set_key] = {
                        action: 1.0 / len(legal_actions_list) for action in legal_actions_list
                    }
                else: # Should be a terminal state or error
                    average_strategy_profile[info_set_key] = {}
        return average_strategy_profile

    def run_self_play(self, iterations=60000):
        print(f"Running self-play for {iterations} iterations...")
        self.train_iterations(num_iterations=iterations, fixed_opponent_policy=None)
        print("Self-play complete. Calculating average strategy.")
        return self.calculate_average_strategy()
# Implements the AdaptivePokerBot, which uses a Bayesian opponent model
# and incrementally trains a CFR model to adapt its strategy.

import random
from bayes import BayesianOpponentModel
from cfr import CFRTrainer
from utils import complete_policy, ALL_INFOSETS, LEGAL_ACTIONS_AT_INFOSET # For fallback strategy

class AdaptivePokerBot:
    def __init__(self, initial_equilibrium_policy):
        # The bot starts with a baseline strategy (e.g., a pre-computed Nash equilibrium).
        self.base_equilibrium_policy = complete_policy(initial_equilibrium_policy) # Ensure it's complete

        # The Bayesian model to learn the opponent's tendencies.
        self.bayesian_opponent_model = BayesianOpponentModel()

        # The CFR trainer to compute a best response to the opponent model.
        self.response_cfr_trainer = CFRTrainer()
        # Initialize CFR trainer (e.g. by training 0 iterations or loading state if applicable)
        # The original trained with opp_pol = self.eq_policy, iterations = 0.
        # This seems to be just for initializing internal structures or if train had side effects.
        # My CFRTrainer does not require this. It's ready.

        # The current policy the bot uses for acting. Initially, it's the base equilibrium.
        self.current_acting_policy = dict(self.base_equilibrium_policy)

        # Counter for observations before triggering a policy update.
        self.observations_since_last_update = 0
        self.update_batch_size = 10 # Update policy every 10 observations
        self.cfr_update_iterations = 200 # Number of CFR iterations for each update

    # Observes the opponent's action and updates the Bayesian model.
    # Triggers a policy recalculation if enough observations have been made.
    def observe_opponent_action(self, opponent_info_set_key, opponent_action):
        self.bayesian_opponent_model.observe_action(opponent_info_set_key, opponent_action)
        self.observations_since_last_update += 1

        # If batch size is reached, update the bot's policy.
        if self.observations_since_last_update >= self.update_batch_size:
            # print("Adaptive bot updating policy...")
            # Get the opponent's current estimated strategy from the Bayesian model.
            # This needs to be a full policy dictionary.
            estimated_opponent_policy = {}
            for info_key in ALL_INFOSETS: # Iterate over all possible info_sets
                estimated_opponent_policy[info_key] = self.bayesian_opponent_model.get_posterior_strategy(info_key)
            
            # Ensure this estimated policy is complete (covers all infosets).
            # get_posterior_strategy should provide uniform for unobserved infosets if handled well.
            # The original used: opp = {k:self.bayes.posterior(k) for k in INFO_SETS}
            # My bayes.get_posterior_strategy has a fallback for unknown keys.

            # Train the CFR model to find a best response to this estimated opponent policy.
            # Note: The CFR trainer's internal state (regrets, strategy sums) will evolve.
            # We are re-training/incrementally training the same CFR instance.
            self.response_cfr_trainer.train_iterations(
                num_iterations=self.cfr_update_iterations,
                fixed_opponent_policy=complete_policy(estimated_opponent_policy) # Ensure completeness
            )
            
            # The new acting policy is the average strategy from this CFR training.
            self.current_acting_policy = complete_policy(self.response_cfr_trainer.calculate_average_strategy())
            
            # Reset the observation counter.
            self.observations_since_last_update = 0
            # print("Adaptive bot policy updated.")

    # Decides on an action to take at the given information set.
    def choose_action(self, own_info_set_key):
        # Get the strategy for the current info set from the bot's current policy.
        # Fallback to the base equilibrium if the info set is somehow not in the current policy.
        strategy_profile = self.current_acting_policy.get(own_info_set_key)

        if not strategy_profile: # Fallback if info_set_key is missing
            # This could happen if current_acting_policy became incomplete somehow
            # Or if own_info_set_key is for a state not covered by ALL_INFOSETS (error)
            # Default to base equilibrium policy for safety
            strategy_profile = self.base_equilibrium_policy.get(own_info_set_key)
            if not strategy_profile: # Ultimate fallback: uniform random for this specific infoset
                legal_actions_list = LEGAL_ACTIONS_AT_INFOSET.get(own_info_set_key, [])
                if not legal_actions_list: return None # No actions possible (should be terminal)
                return random.choice(legal_actions_list) # Should be weighted choice

        # Ensure strategy_profile is not empty (e.g., for terminal but somehow reached)
        if not strategy_profile:
            legal_actions_list = LEGAL_ACTIONS_AT_INFOSET.get(own_info_set_key, [])
            if not legal_actions_list: return None # Should be terminal
            return random.choice(legal_actions_list)


        # Unpack actions and their probabilities from the strategy profile.
        actions = list(strategy_profile.keys())
        probabilities = list(strategy_profile.values())

        # Ensure probabilities sum to 1 (or close enough) for random.choices
        if not (0.999 < sum(probabilities) < 1.001) and sum(probabilities) != 0 :
            # Normalize if not summing to 1, can happen due to float precision or incomplete policy part
            # print(f"Warning: Probabilities for {own_info_set_key} do not sum to 1: {sum(probabilities)}. Normalizing.")
            total_prob = sum(probabilities)
            if total_prob == 0: # All probabilities are zero, pick uniformly
                 return random.choice(actions) if actions else None
            probabilities = [p / total_prob for p in probabilities]


        # Randomly choose an action based on the probabilities in the strategy.
        if not actions: return None # Should not happen for a valid infoset
        chosen_action = random.choices(actions, weights=probabilities, k=1)[0]
        return chosen_action
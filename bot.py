# Implements the AdaptivePokerBot, which uses a Bayesian opponent model
# and incrementally trains a CFR model to adapt its strategy.

import random
from bayes import BayesianOpponentModel
from cfr import CFRTrainer
from utils import complete_policy, ALL_INFOSETS, LEGAL_ACTIONS_AT_INFOSET # For fallback strategy

class AdaptivePokerBot:
    def __init__(self, initial_equilibrium_policy, prior_belief=None):
        # The bot starts with a baseline strategy (e.g., a pre-computed Nash equilibrium).
        self.base_equilibrium_policy = complete_policy(initial_equilibrium_policy) # Ensure it's complete

        # The Bayesian model to learn the opponent's tendencies.
        self.bayesian_opponent_model = BayesianOpponentModel(prior_belief)

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
        self.cfr_update_iterations = 1 # Number of CFR iterations for each update

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
        strategy_profile = self.current_acting_policy.get(own_info_set_key)
        # print(f"Adaptive bot strategy at {own_info_set_key}: {strategy_profile}")
        # Unpack actions and their probabilities from the strategy profile.
        actions = list(strategy_profile.keys())
        probabilities = list(strategy_profile.values())

        # # Randomly choose an action based on the probabilities in the strategy.
        # if not actions: return None # Should not happen for a valid infoset
        chosen_action = random.choices(actions, weights=probabilities, k=1)[0]
        return chosen_action
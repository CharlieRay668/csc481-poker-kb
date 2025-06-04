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

        # The current policy the bot uses for acting. Initially, it's nash.
        self.current_acting_policy = dict(self.base_equilibrium_policy)

        # Counter for observations before triggering a policy update.
        self.observations_since_last_update = 0
        self.update_batch_size = 50 # Update policy every 10 observations
        self.cfr_update_iterations = 200 # Number of CFR iterations for each update
        self.prior_strength_for_cfr = 40000
        self._seed_cfr_with_base_policy()

    def _seed_cfr_with_base_policy(self):
        """Seeds the CFR trainer's strategy sum with the base equilibrium policy."""
        self.response_cfr_trainer.seed_strategy_sum(
            self.base_equilibrium_policy, 
            self.prior_strength_for_cfr
        )
        # Regrets should be cleared before new BR training starts.
        # train_iterations will handle this if reset_regrets_before_training=True.

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
                inferred_op_policy=complete_policy(estimated_opponent_policy)
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
    
    def get_opponent_belief(self):
        """
        Returns the current belief about the opponent's strategy at the given information set.
        This is useful for debugging or understanding the bot's internal state.
        """
        return self.bayesian_opponent_model.get_full_posterior_policy()
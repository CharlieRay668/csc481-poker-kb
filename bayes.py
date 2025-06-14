from collections import defaultdict
from utils import LEGAL_ACTIONS_AT_INFOSET, ALL_INFOSETS 

class BayesianOpponentModel:
    def __init__(self, prior_belief=None):
        # Initialize alpha parameters for the Dirichlet distribution.
        # For each information set, and for each legal action in that set,
        self.alpha_counts = {}
        self.prior_belief = prior_belief
        self.learning_rate = 2
        if self.prior_belief is not None:
            # If a prior belief is provided, use it to initialize alpha counts.
            # This should be a dictionary mapping info set keys to action counts.
            for info_set_key, action_counts in self.prior_belief.items():
                # scale action counts by 10 to give more emphasis to prior belief
                action_counts = {action: count * 0.1 for action, count in action_counts.items()}
                self.alpha_counts[info_set_key] = action_counts
        else:
            # If no prior belief is provided, initialize all info sets with a small count for each legal action.
            # This ensures that we have a non-zero prior for all actions at all info sets.
            for info_set_key in ALL_INFOSETS: # Iterate over all info sets
                legal_actions_list = LEGAL_ACTIONS_AT_INFOSET.get(info_set_key, [])
                if legal_actions_list: # Only for infosets where actions are possible
                    self.alpha_counts[info_set_key] = {action: 0.1 for action in legal_actions_list}

    # Observes an action taken by the opponent at a given information set and updates the model.
    def observe_action(self, info_set_key, observed_action):
        # Increment the count for the observed action at this information set.
        # This is the core of the Dirichlet update.
        if observed_action in self.alpha_counts[info_set_key]:
            self.alpha_counts[info_set_key][observed_action] += self.learning_rate

    # Calculates the posterior probability distribution over actions for a given information set.
    # This is the opponent's estimated strategy based on observations.
    def get_posterior_strategy(self, info_set_key):
        current_counts = self.alpha_counts[info_set_key]
        total_counts = sum(current_counts.values())

        # The posterior probability of an action is its count divided by the total counts for that info set.
        if total_counts == 0: # Avoid division by zero if all counts are somehow zero (e.g. after init with 0)
            num_legal_actions = len(current_counts)
            if num_legal_actions == 0: return {}
            print("Warning: Total counts for info set", info_set_key, "is zero. Returning uniform strategy.")
            return {action: 1.0 / num_legal_actions for action in current_counts}

        posterior_strategy = {action: count / total_counts for action, count in current_counts.items()}
        return posterior_strategy
    
    def get_full_posterior_policy(self):
        """
        Returns the full posterior policy for all information sets.
        This is useful for getting a complete view of the opponent's estimated strategy.
        """
        full_policy = {}
        for info_set_key in ALL_INFOSETS:
            full_policy[info_set_key] = self.get_posterior_strategy(info_set_key)
        # return a copy
        full_policy = {k: v.copy() for k, v in full_policy.items()}
        return full_policy
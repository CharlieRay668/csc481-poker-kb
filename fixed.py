# This file provides functions to create fixed (non-learning) opponent policies,
# such as a uniformly random policy or an always-raise aggressive policy.

from utils import LEGAL_ACTIONS_AT_INFOSET, ALL_INFOSETS

# Creates a fixed policy based on a specified style.
# 'uniform': Plays actions with uniform probability at each information set.
# 'aggressive': Always raises if possible; otherwise, checks/calls.
def create_fixed_policy(style_name):
    policy = {}
    # Iterate over all known information sets
    for info_set_key, legal_actions_list in LEGAL_ACTIONS_AT_INFOSET.items():
        if not legal_actions_list: # Should only be for terminal states, not actual infosets
            policy[info_set_key] = {}
            continue

        if style_name == 'uniform':
            # Assign equal probability to all legal actions
            action_probability = 1.0 / len(legal_actions_list)
            policy[info_set_key] = {action: action_probability for action in legal_actions_list}
        elif style_name == 'aggressive':
            # Prefer 'raise' if available
            if 'r' in legal_actions_list:
                policy[info_set_key] = {action: (1.0 if action == 'r' else 0.0) for action in legal_actions_list}
            # If 'raise' is not legal, prefer 'call' (which covers 'check' if no bet)
            elif 'c' in legal_actions_list: # 'c' is usually always available if 'r' isn't
                 policy[info_set_key] = {action: (1.0 if action == 'c' else 0.0) for action in legal_actions_list}
            else: # Fallback if only 'fold' is available (e.g. after max raises and facing a bet)
                  # Or if somehow 'c' is not there (shouldn't be the case if not terminal)
                  # The original had 'c' as the aggressive fallback.
                  # If only 'f' is available, it must be chosen.
                  if 'f' in legal_actions_list and len(legal_actions_list) == 1:
                       policy[info_set_key] = {'f': 1.0}
                  else: # Default to uniform if logic is complex (e.g. 'c', 'f' avail, aggressive might call)
                       action_probability = 1.0 / len(legal_actions_list)
                       policy[info_set_key] = {action: action_probability for action in legal_actions_list}

        else:
            raise ValueError(f"Unknown fixed policy style: {style_name}")
            
    return policy

def copy_policy(policy):
    """
    Creates a deep copy of the given policy.
    This is useful to ensure that modifications to the copied policy do not affect the original.
    """
    return {info_set: actions.copy() for info_set, actions in policy.items()}

def shift_agressive(policy, alpha=0.1):
    """
    Shifts any policy towards a more aggressive stance.
    Moves alpha probability mass towards 'raise' actions away from 'call' or 'check'.
    Does not modify probabilities if 'raise' is not a legal action at the info set.
    Does not modify fold actions.
    """
    new_policy = copy_policy(policy)
    for info_set, actions in new_policy.items():
        if 'r' in actions:  # Only shift if 'raise' is a legal action
            if 'c' in actions:
                # Shift alpha probability mass from 'call' to 'raise'
                actions['r'] += alpha * actions['c']
                actions['c'] -= alpha * actions['c']

            # Normalize the probabilities to ensure they sum to 1
            total_prob = sum(actions.values())
            for action in actions:
                actions[action] /= total_prob

    return new_policy

def shift_passive(policy, alpha=0.1):
    """
    Shifts any policy towards a more passive stance.
    Moves alpha probability mass towards 'call' or 'check' actions away from 'raise'.
    Does not modify probabilities if 'raise' is not a legal action at the info set.
    Does not modify fold actions.
    """
    new_policy = copy_policy(policy)
    for info_set, actions in new_policy.items():
        if 'r' in actions:  # Only shift if 'raise' is a legal action
            if 'c' in actions:
                # Shift alpha probability mass from 'raise' to 'call'
                actions['c'] += alpha * actions['r']
                actions['r'] -= alpha * actions['r']

            # Normalize the probabilities to ensure they sum to 1
            total_prob = sum(actions.values())
            for action in actions:
                actions[action] /= total_prob

    return new_policy


def shift_loose(policy, alpha=0.1):
    """
    Shifts any policy towards a looser stance.
    Moves alpha probability mass towards 'call' or 'check' actions away from 'fold'.
    Does not modify probabilities if 'fold' is not a legal action at the info set.
    """
    new_policy = copy_policy(policy)
    for info_set, actions in new_policy.items():
        if 'f' in actions:  # Only shift if 'fold' is a legal action
            if 'c' in actions:
                # Shift alpha probability mass from 'fold' to 'call'
                actions['c'] += alpha * actions['f']
                actions['f'] -= alpha * actions['f']

            # Normalize the probabilities to ensure they sum to 1
            total_prob = sum(actions.values())
            for action in actions:
                actions[action] /= total_prob

    return new_policy

def shift_tight(policy, alpha=0.1):
    """
    Shifts any policy towards a tighter stance.
    Moves alpha probability mass towards 'fold' actions away from 'call' or 'check'.
    Does not modify probabilities if 'fold' is not a legal action at the info set.
    """
    new_policy = copy_policy(policy)
    for info_set, actions in new_policy.items():
        if 'f' in actions:  # Only shift if 'fold' is a legal action
            if 'c' in actions:
                # Shift alpha probability mass from 'call' to 'fold'
                actions['f'] += alpha * actions['c']
                actions['c'] -= alpha * actions['c']

            # Normalize the probabilities to ensure they sum to 1
            total_prob = sum(actions.values())
            for action in actions:
                actions[action] /= total_prob

    return new_policy
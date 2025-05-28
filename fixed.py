# fixed.py
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
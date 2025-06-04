# This is the main driver script for running Leduc Poker simulations.
# It sets up the adaptive bot, defines opponents, runs simulations, and plots results.
import json 
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from cfr import CFRTrainer
from fixed import create_fixed_policy, shift_loose, shift_agressive, shift_passive, shift_tight
from bot import AdaptivePokerBot
from game import play_one_hand
from utils import complete_policy, ALL_INFOSETS, LEGAL_ACTIONS_AT_INFOSET


def calculate_kl_divergence(policy1, policy2,  epsilon = 1e-9):
    kl_divergence_total = 0.0

    for infoset_key in ALL_INFOSETS:
        # Get strategies for the current infoset from both policies
        # Default to an empty dict if infoset is missing (though policies should be complete)
        strat1 = policy1.get(infoset_key, {})
        strat2_orig = policy2.get(infoset_key, {})

        # Get legal actions for this infoset
        legal_actions = LEGAL_ACTIONS_AT_INFOSET.get(infoset_key)
        if not legal_actions:
            # if infoset_key in strat1 or infoset_key in strat2_orig:
                # print(f"Warning: Infoset '{infoset_key}' has strategies but no legal actions defined. Skipping.")
            continue # Skip if no legal actions defined for this infoset

        # Create a smoothed version of policy2's strategy for this infoset
        strat2_smoothed = {}
        # sum_strat2_probs_before_smoothing = sum(strat2_orig.get(action, 0.0) for action in legal_actions) # For renormalization

        # Ensure all legal actions are present in strat2_smoothed, applying epsilon
        # This handles cases where an action is in strat1 but not explicitly in strat2_orig
        temp_sum_strat2_smoothed = 0.0
        for action in legal_actions:
            prob_q = strat2_orig.get(action, 0.0)
            strat2_smoothed[action] = prob_q + epsilon
            temp_sum_strat2_smoothed += strat2_smoothed[action]
        
        # Renormalize strat2_smoothed so probabilities sum to 1
        if temp_sum_strat2_smoothed > 0:
            for action in legal_actions:
                strat2_smoothed[action] /= temp_sum_strat2_smoothed
        else: # All original probabilities were 0, and epsilon smoothing led to a sum of N*epsilon
            # This case is tricky. If all strat2_orig were 0, it's like a uniform dist after epsilon.
            # If legal_actions is not empty, re-assign uniform distribution.
            if legal_actions:
                uniform_prob = 1.0 / len(legal_actions)
                for action in legal_actions:
                    strat2_smoothed[action] = uniform_prob
            else: # No legal actions, this infoset shouldn't contribute
                continue


        # Calculate KL divergence for this specific infoset
        kl_divergence_infoset = 0.0
        for action in legal_actions:
            prob_p = strat1.get(action, 0.0) # Probability of action in policy1

            if prob_p > 0: # Only consider terms where P(x) > 0
                prob_q_smoothed = strat2_smoothed.get(action)
                if prob_q_smoothed is None or prob_q_smoothed <= 0:
                    return float('inf') 
                
                kl_divergence_infoset += prob_p * math.log(prob_p / prob_q_smoothed)
        
        kl_divergence_total += kl_divergence_infoset
        
    return kl_divergence_total

def save_nash_policy(policy, filename='nash_policy.json'):
    """
    Saves the Nash policy to a JSON file.
    This is useful for later use or analysis.
    """
    with open(filename, 'w') as f:
        # keys are stored as (int, card, card, history) convert to string
        # This ensures that the keys are JSON serializable.
        policy = {"-".join(map(str, key)): value for key, value in policy.items()}
        json.dump(policy, f, indent=4)
        # raise NotImplementedError("Saving Nash policy is not implemented yet.")

def load_nash_policy(filename='nash_policy.json'):
    """
    Loads a Nash policy from a JSON file.
    This is useful for reusing pre-computed policies.
    """
    # check if file exists
    if not os.path.exists(filename):
        return None 
    with open(filename, 'r') as f:
        policy = json.load(f)
        # Convert keys back to tuple format
        # Assuming keys are stored as strings in the format "int-card-card-history"
        parsed_policy = {}
        for key, value in policy.items():
            # Split the key by '-' and convert to tuple
            parts = key.split('-')
            if len(parts) == 3:
                # Convert first element to int, rest are strings
                part_2 = parts[1] if parts[1] != 'None' else None
                key = (parts[0], part_2, parts[2])
            elif len(parts) == 2:
                # If history is empty, we can assume it was not included in the key
                part_2 = parts[1] if parts[1] != 'None' else None
                key = (parts[0], part_2, '')
            else:
                raise ValueError(f"Unexpected key format: {key}")

            parsed_policy[key] = value
    return parsed_policy

# Runs a simulation of the adaptive bot playing against a specified opponent.
# opponent_style: 'uniform', 'aggressive', or 'nash'.
# num_hands: The number of hands to simulate.
# nash_policy_for_setup: The Nash equilibrium policy, used if opponent_style is 'nash'
#                        and also as the initial policy for the AdaptivePokerBot.
def run_simulation(player_strat, opponent_style_name, opponent_policy, num_hands=1000, prior_belief=None, adaptive_update=True, verbose=False):
    if verbose:
        print(f"Starting simulation: AdaptiveBot vs {opponent_style_name} opponent for {num_hands} hands.")

    adaptive_bot = AdaptivePokerBot(initial_equilibrium_policy=player_strat, prior_belief=prior_belief)

    # Track the adaptive bot's bankroll over time.
    # Start with an arbitrary initial bankroll (e.g., 0, as we track changes).
    # The original started with 1000, let's use 0 and plot cumulative winnings.
    bankroll_history = [0.0] # Initial bankroll before any hands
    opponent_policy_history = []
    wins = 0
    for hand_num in range(num_hands):
        if hand_num % (num_hands // 10) == 0 and hand_num > 0 and verbose : # Print progress
             print(f"  Simulating hand {hand_num}/{num_hands} against {opponent_style_name}...")
        # Play one hand and get the payoff for the adaptive bot.
        payoff_for_adaptive_bot, history, op_poly = play_one_hand(adaptive_bot, opponent_policy, adaptive_update=adaptive_update)
        # print(payoff_for_adaptive_bot, history)
        # if payoff_for_adaptive_bot > 0:
        #     wins += 1
        bankroll_history.append(bankroll_history[-1] + payoff_for_adaptive_bot)
        opponent_policy_history.append(op_poly)
    
    if verbose:
        print(f"Simulation vs {opponent_style_name} complete. Final bankroll change: {bankroll_history[-1]}")
    return bankroll_history, opponent_policy_history


def simulate_nash_loose_opponents(nash_policy, num_hands=1000, adaptive_update=True, alpha=0.1):
    """
    Simulates the adaptive bot against various fixed opponent styles.
    Returns a dictionary with the results for each opponent type.
    """
    results = {}
    loose_nash = shift_loose(nash_policy, alpha=alpha)
    opponent_name = f'loose-alpha-{alpha}'
    # Run the simulation against the loose opponent
    bankroll_history, estimated_op_strat = run_simulation(
        player_strat=nash_policy,
        opponent_style_name=opponent_name,
        opponent_policy=loose_nash,
        num_hands=num_hands,
        prior_belief=nash_policy,
        adaptive_update=adaptive_update
    )

    kl_divs = []
    for estimated_op_strat in estimated_op_strat:
        if estimated_op_strat is None:
            continue
        # Compare KL divergence between the estimated strategy and the tight strategy
        kl_divs.append(calculate_kl_divergence(estimated_op_strat, loose_nash))

    results[opponent_name] = {"hist": bankroll_history, "kl_divs": kl_divs}
    
    return results
def siulate_nash_tight_opponents(nash_policy, num_hands=1000, adaptive_update=True, alpha=0.1):
    """
    Simulates the adaptive bot against various fixed opponent styles.
    Returns a dictionary with the results for each opponent type.
    """
    results = {}
        
    tight_nash = shift_tight(nash_policy, alpha=alpha)
    opponent_name = f'tight-alpha-{alpha}'
    # Run the simulation against the tight opponent
    bankroll_history, estimated_op_strat = run_simulation(
        player_strat=nash_policy,
        opponent_style_name=opponent_name,
        opponent_policy=tight_nash,
        num_hands=num_hands,
        prior_belief=nash_policy,
        adaptive_update=adaptive_update
    )

    kl_divs = []
    for estimated_op_strat in estimated_op_strat:
        if estimated_op_strat is None:
            continue
        # Compare KL divergence between the estimated strategy and the tight strategy
        kl_divs.append(calculate_kl_divergence(estimated_op_strat, tight_nash))

    results[opponent_name] = {"hist": bankroll_history, "kl_divs": kl_divs}
    
    return results

def simulate_nash_aggressive_opponents(nash_policy, num_hands=1000, adaptive_update=True, alpha=0.1):
    """
    Simulates the adaptive bot against various aggressive opponent styles.
    Returns a dictionary with the results for each opponent type.
    """
    results = {}
    
    aggressive_nash = shift_agressive(nash_policy, alpha=alpha)
    opponent_name = f'aggressive-alpha-{alpha}'
    # Run the simulation against the aggressive opponent
    bankroll_history, estimated_op_strat = run_simulation(
        player_strat=nash_policy,
        opponent_style_name=opponent_name,
        opponent_policy=aggressive_nash,
        num_hands=num_hands,
        prior_belief=nash_policy,
        adaptive_update=adaptive_update
    )

    kl_divs = []
    for estimated_op_strat in estimated_op_strat:
        if estimated_op_strat is None:
            continue
        # Compare KL divergence between the estimated strategy and the tight strategy
        kl_divs.append(calculate_kl_divergence(estimated_op_strat, aggressive_nash))

    results[opponent_name] = {"hist": bankroll_history, "kl_divs": kl_divs}
    
    return results

def simulate_nash_passive_opponents(nash_policy, num_hands=1000, adaptive_update=True, alpha=0.1):
    """
    Simulates the adaptive bot against various passive opponent styles.
    Returns a dictionary with the results for each opponent type.
    """
    results = {}
    # alphas is range from 0.1 to 0.9 in steps of 0.1
    
    passive_nash = shift_passive(nash_policy, alpha=alpha)
    opponent_name = f'passive-alpha-{alpha}'
    # Run the simulation against the passive opponent
    bankroll_history, estimated_op_strat = run_simulation(
        player_strat=nash_policy,
        opponent_style_name=opponent_name,
        opponent_policy=passive_nash,
        num_hands=num_hands,
        prior_belief=nash_policy,
        adaptive_update=adaptive_update
    )

    kl_divs = []
    for estimated_op_strat in estimated_op_strat:
        if estimated_op_strat is None:
            continue
        # Compare KL divergence between the estimated strategy and the tight strategy
        kl_divs.append(calculate_kl_divergence(estimated_op_strat, passive_nash))

    results[opponent_name] = {"hist": bankroll_history, "kl_divs": kl_divs}
    
    return results

def simulate_uniform(nash_policy, num_hands=1000, adaptive_update=True, alpha=None):
    """
    Simulates the adaptive bot against a uniform opponent.
    Returns a dictionary with the results for the uniform opponent.
    """
    results = {}
    uniform_strat = create_fixed_policy('uniform')
    
    # Run the simulation against the uniform opponent
    bankroll_history, estimated_op_strat = run_simulation(
        player_strat=nash_policy,
        opponent_style_name='uniform',
        opponent_policy=uniform_strat,
        num_hands=num_hands,
        prior_belief=nash_policy,
        adaptive_update=adaptive_update
    )
    
    kl_divs = []
    for estimated_op_strat in estimated_op_strat:
        if estimated_op_strat is None:
            continue
        # Compare KL divergence between the estimated strategy and the tight strategy
        kl_divs.append(calculate_kl_divergence(estimated_op_strat, uniform_strat))

    results["uniform"] = {"hist": bankroll_history, "kl_divs": kl_divs}

    return results

def simulate_aggressive(nash_policy, num_hands=1000, adaptive_update=True, alpha=None):
    """
    Simulates the adaptive bot against an aggressive opponent.
    Returns a dictionary with the results for the aggressive opponent.
    """
    results = {}
    aggressive_strat = create_fixed_policy('aggressive')
    
    # Run the simulation against the aggressive opponent
    bankroll_history, estimated_op_strat = run_simulation(
        player_strat=nash_policy,
        opponent_style_name='aggressive',
        opponent_policy=aggressive_strat,
        num_hands=num_hands,
        prior_belief=nash_policy,
        adaptive_update=adaptive_update
    )

    kl_divs = []
    for estimated_op_strat in estimated_op_strat:
        if estimated_op_strat is None:
            continue
        # Compare KL divergence between the estimated strategy and the tight strategy
        kl_divs.append(calculate_kl_divergence(estimated_op_strat, aggressive_strat))

    results["aggressive"] = {"hist": bankroll_history, "kl_divs": kl_divs}

    return results


def against_self_play(nash_policy, num_hands=1000, adaptive_update=True):
    """
    Simulates the adaptive bot playing against itself using the Nash policy.
    This is useful for testing the bot's performance against a known strategy.
    """
    results = {}
    # The opponent is also the adaptive bot using the same Nash policy
    op_poly = nash_policy
    
    # Run the simulation
    bankroll_history, estimated_op_strat = run_simulation(
        player_strat=nash_policy,
        opponent_style_name='self-play',
        opponent_policy=op_poly,
        num_hands=num_hands,
        prior_belief=nash_policy,
        adaptive_update=adaptive_update
    )
    
    kl_divs = []
    for estimated_op_strat in estimated_op_strat:
        if estimated_op_strat is None:
            continue
        # Compare KL divergence between the estimated strategy and the tight strategy
        kl_divs.append(calculate_kl_divergence(estimated_op_strat, nash_policy))

    results["nash"] = {"hist": bankroll_history, "kl_divs": kl_divs}

    return results

def simulate_plot_and_save(num_simulation_hands, num_trials, simulation_function, sim_name, alpha=None):
    all_trials_adaptive_results = []
    all_trials_nash_results = []

    for trial in range(num_trials):
        print(f"  Trial {trial + 1}/{num_trials} for {sim_name}...")
        current_trial_adaptive = simulation_function(
            complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=True, alpha=alpha
        )
        all_trials_adaptive_results.append(current_trial_adaptive)
        current_trial_nash = simulation_function(
            complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=False, alpha=alpha
        )
        all_trials_nash_results.append(current_trial_nash)


    opponent_names = list(all_trials_adaptive_results[0].keys())
    
    avg_adaptive_bankrolls = {name: [] for name in opponent_names}
    avg_nash_bankrolls = {name: [] for name in opponent_names}
    avg_adaptive_kl_divs = {name: [] for name in opponent_names}

    for opp_name in opponent_names:
        adaptive_bankroll_series = [trial_res[opp_name]["hist"] for trial_res in all_trials_adaptive_results if opp_name in trial_res and "hist" in trial_res[opp_name]]
        nash_bankroll_series = [trial_res[opp_name]["hist"] for trial_res in all_trials_nash_results if opp_name in trial_res and "hist" in trial_res[opp_name]]

        if adaptive_bankroll_series:
            min_len_b_a = min(len(s) for s in adaptive_bankroll_series)
            avg_adaptive_bankrolls[opp_name] = np.mean(np.array([s[:min_len_b_a] for s in adaptive_bankroll_series]), axis=0).tolist()
        if nash_bankroll_series:
            min_len_b_n = min(len(s) for s in nash_bankroll_series)
            avg_nash_bankrolls[opp_name] = np.mean(np.array([s[:min_len_b_n] for s in nash_bankroll_series]), axis=0).tolist()

        if "kl_divs" in all_trials_adaptive_results[0].get(opp_name, {}):
            adaptive_kl_series = [trial_res[opp_name]["kl_divs"] for trial_res in all_trials_adaptive_results if opp_name in trial_res and "kl_divs" in trial_res[opp_name] and trial_res[opp_name]["kl_divs"]]
            if adaptive_kl_series:
                min_len_kl = min(len(kl_list) for kl_list in adaptive_kl_series if kl_list)
                if min_len_kl > 0:
                    kl_series_padded = [kl_list[:min_len_kl] for kl_list in adaptive_kl_series if len(kl_list) >= min_len_kl]
                    if kl_series_padded:
                       avg_adaptive_kl_divs[opp_name] = np.mean(np.array(kl_series_padded), axis=0).tolist()

    # Plot Bankrolls
    plt.figure(figsize=(14, 8))
    plot_title_suffix = f" (Alpha: {alpha})" if alpha is not None else ""
    plt.title(f'Bot Performance: {sim_name}{plot_title_suffix} ({num_trials} Trials, {num_simulation_hands} Hands)')
    colors = plt.get_cmap('tab10', len(opponent_names) * 2)

    for i, opp_name in enumerate(opponent_names):
        for trial_idx in range(num_trials):
            if opp_name in all_trials_adaptive_results[trial_idx] and all_trials_adaptive_results[trial_idx][opp_name].get("hist"):
                plt.plot(all_trials_adaptive_results[trial_idx][opp_name]["hist"], color=colors(i*2), alpha=0.1)
        if avg_adaptive_bankrolls.get(opp_name):
            plt.plot(avg_adaptive_bankrolls[opp_name], label=f'Avg Adaptive vs {opp_name}', color=colors(i*2), linewidth=2)

        for trial_idx in range(num_trials):
            if opp_name in all_trials_nash_results[trial_idx] and all_trials_nash_results[trial_idx][opp_name].get("hist"):
                 plt.plot(all_trials_nash_results[trial_idx][opp_name]["hist"], color=colors(i*2+1), linestyle='--', alpha=0.1)
        if avg_nash_bankrolls.get(opp_name):
            plt.plot(avg_nash_bankrolls[opp_name], label=f'Avg Nash vs {opp_name}', color=colors(i*2+1), linestyle='--', linewidth=2)

    plt.xlabel('Hand Number'); plt.ylabel('Cumulative Payoff')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if not os.path.exists('results'): os.makedirs('results')
    plt.savefig(f'results/{sim_name}_bankroll_alpha_{alpha if alpha is not None else "NA"}.png')
    plt.close() # Close the figure to free memory

    # Plot KL Divergences
    plot_kl = any(avg_adaptive_kl_divs.get(name) for name in opponent_names)
    if plot_kl:
        plt.figure(figsize=(14, 8))
        plt.title(f'Adaptive Bot Opponent Model KL Divergence: {sim_name}{plot_title_suffix}')
        kl_colors = plt.get_cmap('viridis', len(opponent_names))

        for i, opp_name in enumerate(opponent_names):
            if not avg_adaptive_kl_divs.get(opp_name) and not any(trial_res.get(opp_name, {}).get("kl_divs") for trial_res in all_trials_adaptive_results):
                continue
            for trial_idx in range(num_trials):
                if opp_name in all_trials_adaptive_results[trial_idx] and all_trials_adaptive_results[trial_idx][opp_name].get("kl_divs"):
                    kl_data = all_trials_adaptive_results[trial_idx][opp_name]["kl_divs"]
                    if kl_data : plt.plot(range(len(kl_data)), kl_data, color=kl_colors(i), alpha=0.15)
            if avg_adaptive_kl_divs.get(opp_name):
                 plt.plot(range(len(avg_adaptive_kl_divs[opp_name])), avg_adaptive_kl_divs[opp_name], 
                          label=f'Avg KL vs {opp_name}', color=kl_colors(i), linewidth=2)

        plt.xlabel('Hands'); plt.ylabel('KL Divergence (Opponent Model vs True Opponent)')
        plt.grid(True, linestyle=':', alpha=0.6)
        if any(avg_adaptive_kl_divs.get(name) for name in opponent_names):
            plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/{sim_name}_kl_div_alpha_{alpha if alpha is not None else "NA"}.png')
        plt.close() # Close the figure

    # Save Numerical Results
    output_summary = {"sim_params": {"name": sim_name, "alpha": alpha, "hands": num_simulation_hands, "trials": num_trials},
                      "adaptive_perf": {}, "nash_perf": {}, "adaptive_kl": {}}
    for opp_name in opponent_names:
        if avg_adaptive_bankrolls.get(opp_name):
            output_summary["adaptive_perf"][opp_name] = avg_adaptive_bankrolls[opp_name][-1]
        if avg_nash_bankrolls.get(opp_name):
            output_summary["nash_perf"][opp_name] = avg_nash_bankrolls[opp_name][-1]
        if avg_adaptive_kl_divs.get(opp_name) and avg_adaptive_kl_divs[opp_name]:
            output_summary["adaptive_kl"][opp_name] = avg_adaptive_kl_divs[opp_name][-1]
    
    with open(f'results/{sim_name}_summary_alpha_{alpha if alpha is not None else "NA"}.json', 'w') as f:
        json.dump(output_summary, f, indent=2)


# Main execution block
if __name__ == '__main__':
    print('--- Leduc Poker Bot Simulation ---')
    # Fix random seed for reproducibility
    # random.seed(42)

    # 1. Build an approximate Nash Equilibrium strategy using CFR self-play.
    print('Building approximate Nash equilibrium strategy...')
    complete_nash_policy = load_nash_policy('nash_policy.json')
    if complete_nash_policy is None:
        print("No pre-computed Nash policy found. Running CFR self-play to compute it.")
        nash_equilibrium_solver = CFRTrainer()
        start_time = time.time()
        raw_nash_policy = nash_equilibrium_solver.run_self_play(iterations=5_000_000)
        print(f"Nash equilibrium policy computed in {time.time() - start_time:.2f} seconds.")
        # Ensure the computed Nash policy is complete (covers all info sets).
        complete_nash_policy = complete_policy(raw_nash_policy)
        # Save the computed Nash policy for future use.
        save_nash_policy(complete_nash_policy, 'nash_policy.json')
    
    num_simulation_hands = 10000  # Number of hands for each simulation
    num_trials = 10  # Number of trials for averaging results
    alphas = [0.2, 0.4, 0.6, 0.8]  # Alpha values for aggressive/tight/passive strategies
    # 2. Simulate the adaptive bot against various fixed opponent styles.
    # 2.1 Simulate against uniform opponents
    print('Simulating against uniform opponents...')
    simulate_plot_and_save(num_simulation_hands, num_trials, simulate_uniform, 'uniform_opponents')
    # 2.2 Simulate against aggressive opponents
    print('Simulating against ultra aggressive opponents...')
    simulate_plot_and_save(num_simulation_hands, num_trials, simulate_aggressive, 'ultra_aggressive_opponents')
    # 2.3 Simulate against loose opponents
    print('Simulating against loose opponents...')
    for alpha in alphas:
        simulate_plot_and_save(num_simulation_hands, num_trials, simulate_nash_loose_opponents, f'loose_opponents_{alpha}', alpha=alpha)
    # # # 2.4 Simulate against tight opponents
    print('Simulating against tight opponents...')
    for alpha in alphas:
            simulate_plot_and_save(num_simulation_hands, num_trials, siulate_nash_tight_opponents, f'tight_opponents_{alpha}', alpha=alpha)
    # # # 2.5 Simulate against passive opponents
    print('Simulating against passive opponents...')
    for alpha in alphas:
        simulate_plot_and_save(num_simulation_hands, num_trials, simulate_nash_passive_opponents, f'passive_opponents_{alpha}', alpha=alpha)
    # # # 2.6 Simulate against aggressive opponents
    print('Simulating against aggressive opponents...')
    for alpha in alphas:
        simulate_plot_and_save(num_simulation_hands, num_trials, simulate_nash_aggressive_opponents, f'aggressive_opponents_{alpha}', alpha=alpha)
    # # 3. Simulate the adaptive bot playing against itself using the Nash policy.
    print('Simulating Adaptive Bot vs Self-Play (Nash Policy)...')
    self_play_results = []

    for i in range(100):
        sf_res = against_self_play(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=False)
        self_play_results.append(sf_res)
        if i % 100 == 0:
            print(f"Self-play trial {i + 1} completed.")
    # Average the self-play results
    avg_bankrolls = []
    bankrolls = [res['nash']['hist'] for res in self_play_results]
    print(len(bankrolls))
    for i, bankroll in enumerate(bankrolls):
        avg = sum(b[i] for b in bankrolls) / len(bankrolls)
        avg_bankrolls.append(avg)
    print(len(avg_bankrolls))
    # Plot the self-play results
    plt.figure(figsize=(12, 7))
    plt.plot(avg_bankrolls, label=f'Self-Play vs Nash', linewidth=2)
    for bankroll in bankrolls:
        plt.plot(bankroll, color='blue', alpha=0.1)
    plt.xlabel('Hand Number')
    plt.ylabel('Cumulative Payoff (Bankroll Change)')
    plt.title('Adaptive Bot Performance vs Self-Play (Nash Policy)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping
    plt.show()
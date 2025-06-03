# This is the main driver script for running Leduc Poker simulations.
# It sets up the adaptive bot, defines opponents, runs simulations, and plots results.
import json 
import os
import time
import matplotlib.pyplot as plt
import random
from cfr import CFRTrainer
from fixed import create_fixed_policy, shift_loose, shift_agressive, shift_passive, shift_tight
from bot import AdaptivePokerBot
from game import play_one_hand
from utils import complete_policy, ALL_INFOSETS


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
    wins = 0
    for hand_num in range(num_hands):
        if hand_num % (num_hands // 10) == 0 and hand_num > 0 and verbose : # Print progress
             print(f"  Simulating hand {hand_num}/{num_hands} against {opponent_style_name}...")
        # Play one hand and get the payoff for the adaptive bot.
        payoff_for_adaptive_bot, history = play_one_hand(adaptive_bot, opponent_policy, adaptive_update=adaptive_update)
        # print(payoff_for_adaptive_bot, history)
        if payoff_for_adaptive_bot > 0:
            wins += 1
        bankroll_history.append(bankroll_history[-1] + payoff_for_adaptive_bot)
    
    if verbose:
        print(f"Simulation vs {opponent_style_name} complete. Final bankroll change: {bankroll_history[-1]}")
    return bankroll_history


def simulate_nash_loose_opponents(nash_policy, num_hands=1000, adaptive_update=True, alpha=0.1):
    """
    Simulates the adaptive bot against various fixed opponent styles.
    Returns a dictionary with the results for each opponent type.
    """
    results = {}
    loose_nash = shift_loose(nash_policy, alpha=alpha)
    opponent_name = f'loose-alpha-{alpha}'
    # Run the simulation against the loose opponent
    bankroll_history = run_simulation(
        player_strat=nash_policy,
        opponent_style_name=opponent_name,
        opponent_policy=loose_nash,
        num_hands=num_hands,
        prior_belief=nash_policy,
        adaptive_update=adaptive_update
    )

    results[opponent_name] = bankroll_history
    
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
    bankroll_history = run_simulation(
        player_strat=nash_policy,
        opponent_style_name=opponent_name,
        opponent_policy=tight_nash,
        num_hands=num_hands,
        prior_belief=nash_policy,
        adaptive_update=adaptive_update
    )

    results[opponent_name] = bankroll_history
    
    return results

def simulate_nash_aggressive_opponents(nash_policy, num_hands=1000, adaptive_update=True, alpha=0.1):
    """
    Simulates the adaptive bot against various aggressive opponent styles.
    Returns a dictionary with the results for each opponent type.
    """
    results = {}
    print(f"Simulating against aggressive opponent with alpha={alpha}...")
    
    aggressive_nash = shift_agressive(nash_policy, alpha=alpha)
    opponent_name = f'aggressive-alpha-{alpha}'
    # Run the simulation against the aggressive opponent
    bankroll_history = run_simulation(
        player_strat=nash_policy,
        opponent_style_name=opponent_name,
        opponent_policy=aggressive_nash,
        num_hands=num_hands,
        prior_belief=nash_policy,
        adaptive_update=adaptive_update
    )

    results[opponent_name] = bankroll_history
    
    return results

def simulate_nash_passive_opponents(nash_policy, num_hands=1000, adaptive_update=True, alpha=0.1):
    """
    Simulates the adaptive bot against various passive opponent styles.
    Returns a dictionary with the results for each opponent type.
    """
    results = {}
    # alphas is range from 0.1 to 0.9 in steps of 0.1
    print(f"Simulating against passive opponent with alpha={alpha}...")
    
    passive_nash = shift_passive(nash_policy, alpha=alpha)
    opponent_name = f'passive-alpha-{alpha}'
    # Run the simulation against the passive opponent
    bankroll_history = run_simulation(
        player_strat=nash_policy,
        opponent_style_name=opponent_name,
        opponent_policy=passive_nash,
        num_hands=num_hands,
        prior_belief=nash_policy,
        adaptive_update=adaptive_update
    )

    results[opponent_name] = bankroll_history
    
    return results

def simulate_uniform(nash_policy, num_hands=1000, adaptive_update=True, alpha=None):
    """
    Simulates the adaptive bot against a uniform opponent.
    Returns a dictionary with the results for the uniform opponent.
    """
    print('Simulating against uniform opponent...')
    
    uniform_strat = create_fixed_policy('uniform')
    
    # Run the simulation against the uniform opponent
    bankroll_history = run_simulation(
        player_strat=nash_policy,
        opponent_style_name='uniform',
        opponent_policy=uniform_strat,
        num_hands=num_hands,
        prior_belief=nash_policy,
        adaptive_update=adaptive_update
    )
    
    return {'uniform': bankroll_history}

def simulate_aggressive(nash_policy, num_hands=1000, adaptive_update=True, prior_belief=None, alpha=None):
    """
    Simulates the adaptive bot against an aggressive opponent.
    Returns a dictionary with the results for the aggressive opponent.
    """
    print('Simulating against aggressive opponent...')

    if prior_belief is None:
        prior_belief = nash_policy
    
    aggressive_strat = create_fixed_policy('aggressive')
    
    # Run the simulation against the aggressive opponent
    bankroll_history = run_simulation(
        player_strat=nash_policy,
        opponent_style_name='aggressive',
        opponent_policy=aggressive_strat,
        num_hands=num_hands,
        prior_belief=nash_policy,
        adaptive_update=adaptive_update
    )
    
    return {'aggressive': bankroll_history}


def against_self_play(nash_policy, num_hands=1000, adaptive_update=True, verbose=False):
    """
    Simulates the adaptive bot playing against itself using the Nash policy.
    This is useful for testing the bot's performance against a known strategy.
    """
    if verbose:
        print(f"Simulating Adaptive Bot vs Self-Play (Nash Policy) for {num_hands} hands.")
    
    # The opponent is also the adaptive bot using the same Nash policy
    opponent_policy = nash_policy
    
    # Run the simulation
    bankroll_history = run_simulation(
        player_strat=nash_policy,
        opponent_style_name='self-play',
        opponent_policy=opponent_policy,
        num_hands=num_hands,
        adaptive_update=adaptive_update
    )
    
    if verbose:
        print(f"Self-play simulation complete. Final bankroll change: {bankroll_history[-1]}")
    return {"nash": bankroll_history}


def simulate_plot_and_save(num_simulation_hands, num_trials, simulation_function, sim_name, alpha=None):
    total_results = []
    total_nash_results = []
    for trial in range(num_trials):
        print("Running trial", trial + 1, "of", num_trials)
        results = simulation_function(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=True, alpha=alpha)
        nash_results = simulation_function(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=False, alpha=alpha)
        total_results.append(results)
        total_nash_results.append(nash_results)
    # Average the results across trials
    avg_results = {}
    avg_nash_results = {}
    for opponent_name in total_results[0].keys():
        avg_results[opponent_name] = [sum(x) / num_trials for x in zip(*[res[opponent_name] for res in total_results])]
        avg_nash_results[opponent_name] = [sum(x) / num_trials for x in zip(*[res[opponent_name] for res in total_nash_results])]
    # Plot the results
    plt.figure(figsize=(12, 7))
    for opponent_name, bankroll_history in avg_results.items():
        plt.plot(bankroll_history, label=f'Adaptive Bot vs {opponent_name}', alpha=0.7)
    for opponent_name, bankroll_history in avg_nash_results.items():
        plt.plot(bankroll_history, label=f'Nash vs {opponent_name}', linestyle='--', alpha=0.7)
    # Plot all intermediate results with transparency
    for trial in range(num_trials):
        for opponent_name in total_results[trial].keys():
            plt.plot(total_results[trial][opponent_name], color='blue', alpha=0.1)
            plt.plot(total_nash_results[trial][opponent_name], color='orange', linestyle='--', alpha=0.1)
    plt.xlabel('Hand Number')
    plt.ylabel('Cumulative Payoff (Bankroll Change)')
    plt.title(f'Adaptive Bot/Nash Performance vs  {sim_name} (Alpha Variations, {num_simulation_hands} Hands Each)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping
    # Save the plot to a file
    plt.savefig(f'results/{sim_name}_results.png')
    plt.show()
    # Print out differences in final bankrolls for the averages
    print(f"--- Results for {sim_name} ---")
    output_results = {}
    output_results["adaptive"] = {}
    output_results["nash"] = {}
    for opponent_name in avg_results.keys():
        adaptive_final_bankroll = avg_results[opponent_name][-1]
        nash_final_bankroll = avg_nash_results[opponent_name][-1]
        output_results["adaptive"][opponent_name] = adaptive_final_bankroll
        output_results["nash"][opponent_name] = nash_final_bankroll
        print(f"Adaptive Bot average final bankroll change against {opponent_name} : {adaptive_final_bankroll:.2f}")
        print(f"Nash average final bankroll change against {opponent_name} : {nash_final_bankroll:.2f}")

    # Save the results to a JSON file
    results_filename = f'results/{sim_name}_results.json'
    with open(results_filename, 'w') as f:
        json.dump(output_results, f, indent=4)


# Main execution block
if __name__ == '__main__':
    print(len(ALL_INFOSETS))
    print('--- Leduc Poker Bot Simulation ---')
    # Fix random seed for reproducibility
    # random.seed(42)

    # 1. Build an approximate Nash Equilibrium strategy using CFR self-play.
    print('Building approximate Nash equilibrium strategy...')
    complete_nash_policy = load_nash_policy('nash_policy.json')
    if complete_nash_policy is None:
        print("No pre-computed Nash policy found. Running CFR self-play to compute it.")
        nash_equilibrium_solver = CFRTrainer()
        # The number of iterations for self-play affects the quality of the Nash approximation.
        # Original used 30,000. Higher is generally better but takes longer.
        # timing info
        start_time = time.time()
        raw_nash_policy = nash_equilibrium_solver.run_self_play(iterations=5_000_000)
        print(f"Nash equilibrium policy computed in {time.time() - start_time:.2f} seconds.")
        # Ensure the computed Nash policy is complete (covers all info sets).
        complete_nash_policy = complete_policy(raw_nash_policy)
        # Save the computed Nash policy for future use.
        save_nash_policy(complete_nash_policy, 'nash_policy.json')
    
    num_simulation_hands = 1000  # Number of hands for each simulation
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
    # 2.4 Simulate against tight opponents
    print('Simulating against tight opponents...')
    for alpha in alphas:
        simulate_plot_and_save(num_simulation_hands, num_trials, siulate_nash_tight_opponents, f'tight_opponents_{alpha}', alpha=alpha)
    # 2.5 Simulate against passive opponents
    print('Simulating against passive opponents...')
    for alpha in alphas:
        simulate_plot_and_save(num_simulation_hands, num_trials, simulate_nash_passive_opponents, f'passive_opponents_{alpha}', alpha=alpha)
    # 2.6 Simulate against aggressive opponents
    print('Simulating against aggressive opponents...')
    for alpha in alphas:
        simulate_plot_and_save(num_simulation_hands, num_trials, simulate_nash_aggressive_opponents, f'aggressive_opponents_{alpha}', alpha=alpha)
    # 3. Simulate the adaptive bot playing against itself using the Nash policy.
    print('Simulating Adaptive Bot vs Self-Play (Nash Policy)...')
    self_play_results = []

    for i in range(1000):
        sf_res = against_self_play(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=False)
        self_play_results.append(sf_res)
        if i % 100 == 0:
            print(f"Self-play trial {i + 1} completed.")
    # Average the self-play results
    avg_self_play_results = {}
    for opponent_name in self_play_results[0].keys():
        avg_self_play_results[opponent_name] = [sum(x) / len(self_play_results) for x in zip(*[res[opponent_name] for res in self_play_results])]
    # Save the self-play results to a file

    # Plot the results
    plt.figure(figsize=(12, 7))
    for opponent_name, bankroll_history in avg_self_play_results.items():
        plt.plot(bankroll_history, label=f'Adaptive Bot vs Nash', alpha=0.7)
    # Plot all intermediate results with transparency
    for trial in range(num_trials):
        for opponent_name in self_play_results[trial].keys():
            plt.plot(self_play_results[trial][opponent_name], color='blue', alpha=0.1)
            plt.plot(self_play_results[trial][opponent_name], color='orange', linestyle='--', alpha=0.1)
    # Plot the self-play results
    # plt.plot(avg_self_play_results, label='Adaptive Bot vs Self-Play (Nash Policy)', color='green')
    plt.xlabel('Hand Number')
    plt.ylabel('Adaptive Bot Cumulative Payoff (Bankroll Change)')
    plt.title(f'Adaptive Bot Performance vs Self-Play (Nash Policy, {num_simulation_hands} Hands Each)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping
    # Save the self-play plot to a file
    plt.savefig('adaptive_bot_self_play.png')
    plt.show()


    # num_trials = 100
    # total_uniform_results = []
    # total_aggressive_results = []
    # total_uniform_results_no_update = []
    # total_aggressive_results_no_update = []

    # for trial in range(num_trials):
    #     uniform_results = simulate_uniform(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=True)
    #     aggressive_results = simulate_aggressive(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=True)
    #     uniform_results_no_update = simulate_uniform(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=False)
    #     aggressive_results_no_update = simulate_aggressive(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=False)
    #     total_uniform_results.append(uniform_results['uniform'])
    #     total_aggressive_results.append(aggressive_results['aggressive'])
    #     total_uniform_results_no_update.append(uniform_results_no_update['uniform'])
    #     total_aggressive_results_no_update.append(aggressive_results_no_update['aggressive'])

    # # Average the results across trials
    # avg_uniform_results = [sum(x) / num_trials for x in zip(*total_uniform_results)]
    # avg_aggressive_results = [sum(x) / num_trials for x in zip(*total_aggressive_results)]
    # avg_uniform_results_no_update = [sum(x) / num_trials for x in zip(*total_uniform_results_no_update)]
    # avg_aggressive_results_no_update = [sum(x) / num_trials for x in zip(*total_aggressive_results_no_update)]
    # # Plot the results
    # plt.figure(figsize=(12, 7))
    # plt.plot(avg_uniform_results, label='Adaptive Bot vs Uniform Opponent (with updates)', color='blue')
    # plt.plot(avg_aggressive_results, label='Adaptive Bot vs Aggressive Opponent (with updates)', color='orange')
    # plt.plot(avg_uniform_results_no_update, label='Adaptive Bot vs Uniform Opponent (no updates)', color='blue', linestyle='--')
    # plt.plot(avg_aggressive_results_no_update, label='Adaptive Bot vs Aggressive Opponent (no updates)', color='orange', linestyle='--')
    # # Plot all intermediate results with transparency
    # for i in range(num_trials):
    #     plt.plot(total_uniform_results[i], color='blue', alpha=0.1)
    #     plt.plot(total_aggressive_results[i], color='orange', alpha=0.1)
    #     plt.plot(total_uniform_results_no_update[i], color='blue', linestyle='--', alpha=0.1)
    #     plt.plot(total_aggressive_results_no_update[i], color='orange', linestyle='--', alpha=0.1)
    # plt.xlabel('Hand Number')
    # plt.ylabel('Adaptive Bot Cumulative Payoff (Bankroll Change)')
    # plt.title(f'Adaptive Bot Performance vs Fixed Opponents (Alpha Variations, {num_simulation_hands} Hands Each)')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()
    # plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping
    # # Save the plot to a file
    # plt.savefig('adaptive_bot_vs_fixed_opponents.png')
    # plt.show()
    # # Print out differences in final bankrolls for the averages
    # print(f"Average final bankroll change against uniform opponent (with updates): {avg_uniform_results[-1]:.2f}")
    # print(f"Average final bankroll change against aggressive opponent (with updates): {avg_aggressive_results[-1]:.2f}")
    # print(f"Average final bankroll change against uniform opponent (no updates): {avg_uniform_results_no_update[-1]:.2f}")
    # print(f"Average final bankroll change against aggressive opponent (no updates): {avg_aggressive_results_no_update[-1]:.2f}")

    # aggressive_strat = create_fixed_policy('aggressive')
    # results = []
    # results.append(simulate_aggressive(complete_nash_policy, num_hands=1000, adaptive_update=True, prior_belief=aggressive_strat))
    # results.append(simulate_aggressive(complete_nash_policy, num_hands=1000, adaptive_update=True))
    # results.append(simulate_aggressive(complete_nash_policy, num_hands=1000, adaptive_update=False))
    # # Plot the results
    # plt.figure(figsize=(12, 7))
    # for i, result in enumerate(results):
    #     for opponent_name, bankroll_history in result.items():
    #         if i == 0:
    #             label = f'Adaptive Bot vs {opponent_name} (with updates and prior knowledge)'
    #         elif i == 1:
    #             label = f'Adaptive Bot vs {opponent_name} (with updates)'
    #         else:
    #             label = f'Adaptive Bot vs {opponent_name} (no updates)'
    #         plt.plot(bankroll_history, label=label, alpha=0.7)
    # plt.xlabel('Hand Number')
    # plt.ylabel('Adaptive Bot Cumulative Payoff (Bankroll Change)')
    # plt.title(f'Adaptive Bot Performance vs Aggressive Opponent (Alpha Variations, {num_simulation_hands} Hands Each)')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()
    # plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping
    # # Save the plot to a file
    # # plt.savefig('adaptive_bot_vs_aggressive_opponent.png')
    # plt.show()

    # sim_nash_uniform_aggressive(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=False)
    # Simulate against loose opponents with varying aggression levels
    # print('Step 4: Simulating against loose opponents with varying aggression levels...')
    # num_trials = 10
    # total_loose_results = []
    # total_loose_results_no_update = []
    # for trial in range(num_trials):
    #     loose_results = simulate_nash_loose_opponents(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=True)
    #     loose_results_no_update = simulate_nash_loose_opponents(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=False)
    #     total_loose_results.append(loose_results)
    #     total_loose_results_no_update.append(loose_results_no_update)
    # # Average the results across trials
    # avg_loose_results = {}
    # avg_loose_results_no_update = {}
    # for opponent_name in total_loose_results[0].keys():
    #     avg_loose_results[opponent_name] = [sum(x) / num_trials for x in zip(*[res[opponent_name] for res in total_loose_results])]
    #     avg_loose_results_no_update[opponent_name] = [sum(x) / num_trials for x in zip(*[res[opponent_name] for res in total_loose_results_no_update])]
    # # Plot the results
    # plt.figure(figsize=(12, 7))
    # for opponent_name, bankroll_history in avg_loose_results.items():
    #     plt.plot(bankroll_history, label=f'Adaptive Bot vs {opponent_name} (with updates)', alpha=0.7)
    # for opponent_name, bankroll_history in avg_loose_results_no_update.items():
    #     plt.plot(bankroll_history, label=f'Adaptive Bot vs {opponent_name} (no updates)', linestyle='--', alpha=0.7)
    # plt.xlabel('Hand Number')
    # plt.ylabel('Adaptive Bot Cumulative Payoff (Bankroll Change)')
    # plt.title(f'Adaptive Bot Performance vs Loose Opponents (Alpha Variations, {num_simulation_hands} Hands Each)')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()
    # plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping
    # # Save the plot to a file
    # # plt.savefig('adaptive_bot_vs_loose_opponents.png')
    # plt.show()
    # loose_opponent_results = simulate_nash_loose_opponents(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=True)
    # loose_opp_res_no_update = simulate_nash_loose_opponents(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=False)
    # print('Loose opponent simulations complete.\n')
    # # Plot upadted results in green, no update in red
    # plt.figure(figsize=(12, 7))
    # for opponent_name, bankroll_history in loose_opponent_results.items():
    #     plt.plot(bankroll_history, label=f'Adaptive Bot vs {opponent_name} (with updates)')
    # for opponent_name, bankroll_history in loose_opp_res_no_update.items():
    #     plt.plot(bankroll_history, label=f'Adaptive Bot vs {opponent_name} (no updates)', linestyle='--')
    # plt.xlabel('Hand Number')
    # plt.ylabel('Adaptive Bot Cumulative Payoff (Bankroll Change)')
    # plt.title(f'Adaptive Bot vs Loose Opponents (Alpha Variations, {num_simulation_hands} Hands Each)')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()
    # plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping
    # # Save the plot to a file
    # plt.savefig('adaptive_bot_vs_loose_opponents.png')
    # plt.show()
    # plot 100 self play interactions:
    # results = []
    # for i in range(100):
    #     res = against_self_play(complete_nash_policy, num_hands=10000, adaptive_update=False)
    #     results.append(res)
    # # Plot the self-play results
    # plt.figure(figsize=(12, 7))
    # for i, bankroll_history in enumerate(results):
    #     plt.plot(bankroll_history, label=f'Self-Play {i+1}', alpha=0.5)
    # plt.xlabel('Hand Number')
    # plt.ylabel('Adaptive Bot Cumulative Payoff (Bankroll Change)')
    # plt.title(f'Adaptive Bot Self-Play Performance (100 Simulations, 10,000 Hands Each)')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # # plt.legend()
    # plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping
    # # Save the plot to a file
    # # plt.savefig('adaptive_bot_self_play.png')
    # plt.show()
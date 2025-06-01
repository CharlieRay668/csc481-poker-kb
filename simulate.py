# This is the main driver script for running Leduc Poker simulations.
# It sets up the adaptive bot, defines opponents, runs simulations, and plots results.
import json 
import os
import time
import matplotlib.pyplot as plt
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
        raise NotImplementedError("Saving Nash policy is not implemented yet.")

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
            if len(parts) == 4:
                # Convert first element to int, rest are strings
                part_2 = parts[2] if parts[2] != 'None' else None
                key = (int(parts[0]), parts[1], part_2, parts[3])
            elif len(parts) == 3:
                # If history is empty, we can assume it was not included in the key
                part_2 = parts[2] if parts[2] != 'None' else None
                key = (int(parts[0]), parts[1], part_2, '')
            else:
                raise ValueError(f"Unexpected key format: {key}")

            parsed_policy[key] = value
    return parsed_policy

# Runs a simulation of the adaptive bot playing against a specified opponent.
# opponent_style: 'uniform', 'aggressive', or 'nash'.
# num_hands: The number of hands to simulate.
# nash_policy_for_setup: The Nash equilibrium policy, used if opponent_style is 'nash'
#                        and also as the initial policy for the AdaptivePokerBot.
def run_simulation(player_strat, opponent_style_name, opponent_policy, num_hands=1000, prior_belief=None, adaptive_update=True):
    print(f"Starting simulation: AdaptiveBot vs {opponent_style_name} opponent for {num_hands} hands.")

    adaptive_bot = AdaptivePokerBot(initial_equilibrium_policy=player_strat, prior_belief=prior_belief)

    # Track the adaptive bot's bankroll over time.
    # Start with an arbitrary initial bankroll (e.g., 0, as we track changes).
    # The original started with 1000, let's use 0 and plot cumulative winnings.
    bankroll_history = [0.0] # Initial bankroll before any hands
    bot_wins = 0
    for hand_num in range(num_hands):
        if hand_num % (num_hands // 10) == 0 and hand_num > 0 : # Print progress
             print(f"  Simulating hand {hand_num}/{num_hands} against {opponent_style_name}...")
        # Play one hand and get the payoff for the adaptive bot.
        payoff_for_adaptive_bot = play_one_hand(adaptive_bot, opponent_policy, adaptive_update=adaptive_update)
        if payoff_for_adaptive_bot % 2 != 0:
            raise ValueError(f"Payoff for adaptive bot is not even: {payoff_for_adaptive_bot}. This should not happen.")
        # if payoff_for_adaptive_bot > 0:
        #     bot_wins += 1
        #     print(f"  Adaptive bot won hand {hand_num + 1} with payoff: {payoff_for_adaptive_bot}")
        # Update the bankroll with the result of the hand.
        bankroll_history.append(bankroll_history[-1] + payoff_for_adaptive_bot)
    
    print(f"Simulation vs {opponent_style_name} complete. Final bankroll change: {bankroll_history[-1]}")
    print(f"Adaptive bot won {bot_wins} out of {num_hands} hands ({bot_wins / num_hands * 100:.2f}%).")
    return bankroll_history


def simulate_nash_loose_opponents(nash_policy, num_hands=1000, adaptive_update=True):
    """
    Simulates the adaptive bot against various fixed opponent styles.
    Returns a dictionary with the results for each opponent type.
    """
    results = {}
    # alphas is range from 0.1 to 0.9 in steps of 0.1
    alphas = [0.1 * i for i in range(1, 10)]  # [0.1, 0.2, ..., 0.9]
    alphas = [0.2 * i for i in range(1, 5)]

    for alpha in alphas:
        print(f"Simulating against loose opponent with alpha={alpha}...")
        
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

def simulate_uniform(nash_policy, num_hands=1000, adaptive_update=True):
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

def simulate_aggressive(nash_policy, num_hands=1000, adaptive_update=True):
    """
    Simulates the adaptive bot against an aggressive opponent.
    Returns a dictionary with the results for the aggressive opponent.
    """
    print('Simulating against aggressive opponent...')
    
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


def against_self_play(nash_policy, num_hands=1000, adaptive_update=True):
    """
    Simulates the adaptive bot playing against itself using the Nash policy.
    This is useful for testing the bot's performance against a known strategy.
    """
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
    
    print(f"Self-play simulation complete. Final bankroll change: {bankroll_history[-1]}")
    return bankroll_history



# Main execution block
if __name__ == '__main__':
    print(len(ALL_INFOSETS))
    print('--- Leduc Poker Bot Simulation ---')

    # 1. Build an approximate Nash Equilibrium strategy using CFR self-play.
    print('Step 1: Building approximate Nash equilibrium strategy...')
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
    
    num_simulation_hands = 10000  # Number of hands for each simulation

    uniform_results = simulate_uniform(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=True)
    aggressive_results = simulate_aggressive(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=True)
    uniform_results_no_update = simulate_uniform(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=False)
    aggressive_results_no_update = simulate_aggressive(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=False)
    # plot
    plt.figure(figsize=(12, 7))
    # Plot uniform results
    for opponent_name, bankroll_history in uniform_results.items():
        plt.plot(bankroll_history, label=f'Adaptive Bot vs {opponent_name} (with updates)')
    for opponent_name, bankroll_history in uniform_results_no_update.items():
        plt.plot(bankroll_history, label=f'Adaptive Bot vs {opponent_name} (no updates)', linestyle='--')
    # Plot aggressive results
    for opponent_name, bankroll_history in aggressive_results.items():
        plt.plot(bankroll_history, label=f'Adaptive Bot vs {opponent_name} (with updates)')
    for opponent_name, bankroll_history in aggressive_results_no_update.items():
        plt.plot(bankroll_history, label=f'Adaptive Bot vs {opponent_name} (no updates)', linestyle='--')
    plt.xlabel('Hand Number')
    plt.ylabel('Adaptive Bot Cumulative Payoff (Bankroll Change)')
    plt.title(f'Adaptive Bot vs Fixed Opponents (Alpha Variations, {num_simulation_hands} Hands Each)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping
    # Save the plot to a file
    # plt.savefig('adaptive_bot_vs_fixed_opponents.png')
    plt.show()


    # sim_nash_uniform_aggressive(complete_nash_policy, num_hands=num_simulation_hands, adaptive_update=False)
    # Simulate against loose opponents with varying aggression levels
    # print('Step 4: Simulating against loose opponents with varying aggression levels...')
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
# simulate.py
# This is the main driver script for running Leduc Poker simulations.
# It sets up the adaptive bot, defines opponents, runs simulations, and plots results.

import matplotlib.pyplot as plt
from cfr import CFRTrainer
from fixed import create_fixed_policy
from bot import AdaptivePokerBot
from game import play_one_hand
from utils import complete_policy

# Runs a simulation of the adaptive bot playing against a specified opponent.
# opponent_style: 'uniform', 'aggressive', or 'nash'.
# num_hands: The number of hands to simulate.
# nash_policy_for_setup: The Nash equilibrium policy, used if opponent_style is 'nash'
#                        and also as the initial policy for the AdaptivePokerBot.
def run_simulation(opponent_style_name, num_hands=1000, nash_policy_for_setup=None):
    print(f"Starting simulation: AdaptiveBot vs {opponent_style_name} opponent for {num_hands} hands.")

    # Define the opponent's policy
    if opponent_style_name == 'nash':
        if nash_policy_for_setup is None:
            raise ValueError("Nash policy must be provided for 'nash' opponent style.")
        # The opponent plays according to the provided Nash equilibrium policy.
        # Ensure it's complete (covers all info sets).
        opponent_policy = complete_policy(nash_policy_for_setup)
    else:
        # Create a fixed policy (e.g., 'uniform' or 'aggressive').
        # This will also be complete by design of create_fixed_policy and LEGAL_ACTIONS_AT_INFOSET.
        opponent_policy = create_fixed_policy(opponent_style_name)

    # Initialize the AdaptivePokerBot. It starts with the Nash equilibrium policy.
    if nash_policy_for_setup is None:
        raise ValueError("Nash policy must be provided to initialize AdaptivePokerBot.")
    adaptive_bot = AdaptivePokerBot(initial_equilibrium_policy=nash_policy_for_setup)

    # Track the adaptive bot's bankroll over time.
    # Start with an arbitrary initial bankroll (e.g., 0, as we track changes).
    # The original started with 1000, let's use 0 and plot cumulative winnings.
    bankroll_history = [0.0] # Initial bankroll before any hands

    for hand_num in range(num_hands):
        if hand_num % (num_hands // 10) == 0 and hand_num > 0 : # Print progress
             print(f"  Simulating hand {hand_num}/{num_hands} against {opponent_style_name}...")
        # Play one hand and get the payoff for the adaptive bot.
        payoff_for_adaptive_bot = play_one_hand(adaptive_bot, opponent_policy)
        # Update the bankroll with the result of the hand.
        bankroll_history.append(bankroll_history[-1] + payoff_for_adaptive_bot)
    
    print(f"Simulation vs {opponent_style_name} complete. Final bankroll change: {bankroll_history[-1]}")
    return bankroll_history


# Main execution block
if __name__ == '__main__':
    print('--- Leduc Poker Bot Simulation ---')

    # 1. Build an approximate Nash Equilibrium strategy using CFR self-play.
    print('Step 1: Building approximate Nash equilibrium strategy...')
    nash_equilibrium_solver = CFRTrainer()
    # The number of iterations for self-play affects the quality of the Nash approximation.
    # Original used 30,000. Higher is generally better but takes longer.
    raw_nash_policy = nash_equilibrium_solver.run_self_play(iterations=30000) # Reduced for faster example if needed
    
    # Ensure the computed Nash policy is complete (covers all info sets).
    comprehensive_nash_policy = complete_policy(raw_nash_policy)
    print('Approximate Nash equilibrium strategy ready.\n')

    # 2. Run simulations against different opponent types.
    print('Step 2: Running simulations (3,000 hands each)...')
    num_simulation_hands = 1000 # As per original plot

    # Simulate against a uniform random opponent
    bankroll_vs_uniform = run_simulation(
        opponent_style_name='uniform',
        num_hands=num_simulation_hands,
        nash_policy_for_setup=comprehensive_nash_policy
    )

    # Simulate against an always-raise (aggressive) opponent
    bankroll_vs_aggressive = run_simulation(
        opponent_style_name='aggressive',
        num_hands=num_simulation_hands,
        nash_policy_for_setup=comprehensive_nash_policy
    )

    # Simulate against an opponent playing the computed Nash equilibrium strategy
    bankroll_vs_nash = run_simulation(
        opponent_style_name='nash',
        num_hands=num_simulation_hands,
        nash_policy_for_setup=comprehensive_nash_policy
    )
    print('All simulations complete.\n')

    # 3. Plot the results.
    print('Step 3: Plotting results...')
    plt.figure(figsize=(12, 7)) # Slightly larger figure for better readability
    
    plt.plot(bankroll_vs_uniform, label='Adaptive Bot vs Uniform Random Opponent')
    plt.plot(bankroll_vs_aggressive, label='Adaptive Bot vs Always-Raise Opponent')
    plt.plot(bankroll_vs_nash, label='Adaptive Bot vs Nash Equilibrium Opponent')
    
    plt.xlabel('Hand Number')
    plt.ylabel('Adaptive Bot Cumulative Payoff (Bankroll Change)')
    plt.title(f'Adaptive CFR-Bayesian Bot Performance ({num_simulation_hands} Hands Each)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout() # Adjusts plot to ensure everything fits without overlapping
    
    # Save the plot to a file
    plot_filename = 'leduc_bot_performance.png'
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    
    # Display the plot
    plt.show()
    print('--- Simulation Finished ---')
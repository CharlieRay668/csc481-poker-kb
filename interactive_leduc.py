import csv
import numpy as np
import rlcard
import random

CARD_MAPPING = ['J', 'Q', 'K']
ACTIONS = ['Call', 'Raise', 'Fold', 'Check']

def parse_state(state):
    hand_card_idx = np.argmax(state[0:3])
    public_card_idx = np.argmax(state[3:6]) if np.max(state[3:6]) > 0 else -1

    hand_card = CARD_MAPPING[hand_card_idx]
    public_card = CARD_MAPPING[public_card_idx] if public_card_idx != -1 else "None"

    player_chips = np.argmax(state[6:21])
    opponent_chips = np.argmax(state[21:])

    return hand_card, public_card, player_chips, opponent_chips


def describe_state(state):
    hand_card, public_card, player_chips, opponent_chips = parse_state(state)
    description = (
        f"Hand Card: {hand_card}, Public Card: {public_card}, Player Chips: {player_chips}, Opponent Chips: {opponent_chips}"
    )
    return description


def main():
    env = rlcard.make('leduc-holdem')
    state, player_id = env.reset()
    done = False

    print("Welcome to Leduc Hold'em!")

    while not done:
        print("Current State:")
        print(describe_state(state['obs']))
        print("Available Actions:")
        for i, action in enumerate(ACTIONS):
            print(f"{i}: {action}")

        action = input("Choose an action (0-3): ")
        try:
            action = int(action)
            if action not in range(4):
                raise ValueError
        except ValueError:
            print("Invalid action. Please choose a valid action (0-3).")
            continue

        next_state, next_player_id = env.step(action)
        done = env.is_over()

        if done:
            payoffs = env.get_payoffs()
            print(f"Game Over! Payoff: {payoffs[player_id]}")
        else:
            print("Action taken: ", ACTIONS[action])

        state = next_state

    print("Thanks for playing!")


if __name__ == "__main__":
    main()
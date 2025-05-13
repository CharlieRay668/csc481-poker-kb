import csv
import numpy as np
import rlcard


env = rlcard.make('leduc-holdem')

state, player_id = env.reset()
next_state, next_player_id = env.step(action)

DATASET_PATH = "leduc_dataset.csv"

CARD_MAPPING = ['J', 'Q', 'K']
ACTIONS = ['Call', 'Raise', 'Fold', 'Check']

def parse_state(state_str):
    state = np.fromstring(state_str.strip('[]'), sep=' ')

    hand_card_idx = np.argmax(state[0:3])
    public_card_idx = np.argmax(state[3:6]) if np.max(state[3:6]) > 0 else -1

    hand_card = CARD_MAPPING[hand_card_idx]
    public_card = CARD_MAPPING[public_card_idx] if public_card_idx != -1 else "None"

    player_chips = np.argmax(state[6:21])
    opponent_chips = np.argmax(state[21:])

    return hand_card, public_card, player_chips, opponent_chips

def describe_row(row):
    state, action, reward, next_state, done = row
    hand_card, public_card, player_chips, opponent_chips = parse_state(state)
    next_hand_card, next_public_card, next_player_chips, next_opponent_chips = parse_state(next_state)

    action_str = ACTIONS[int(action)]
    status = "Game Ended" if done == 'True' else "Game Continues"

    description = (
        f"Current State: Hand Card: {hand_card}, Public Card: {public_card}, Player Chips: {player_chips}, Opponent Chips: {opponent_chips}\n"
        f"Action Taken: {action_str}, Reward: {reward}, {status}\n"
        f"Next State: Hand Card: {next_hand_card}, Public Card: {next_public_card}, Player Chips: {next_player_chips}, Opponent Chips: {next_opponent_chips}\n"
    )
    return description

def main():
    with open(DATASET_PATH, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        x = 0
        for row in reader:
            print(describe_row(row))
            if x == 5:
                break
            x += 1

if __name__ == "__main__":
    main()

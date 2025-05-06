import pytest
from leduc_env import LeducEnv
from bots import AlwaysCallAgent, AlwaysFoldAgent

def play_hand(env, agents):
    obs, legal = env.reset()
    done = False
    while not done:
        pid = env._current_player
        action = agents[pid].step(obs, legal)
        (obs, legal), _, done = env.step(action)

@pytest.mark.parametrize("hands", [1000])
def test_leduc_runs(hands):
    env = LeducEnv(seed=2025)
    agents = [AlwaysCallAgent(), AlwaysFoldAgent()]
    for _ in range(hands):
        play_hand(env, agents)

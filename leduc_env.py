"""
Minimal wrapper around RLCard’s Leduc Hold’em environment
that exposes exactly two calls we need downstream:

    obs, legal = env.observe()
    (next_obs, next_legal), reward, done = env.step(action)
"""
from __future__ import annotations
import rlcard
from typing import Any, Tuple, List


class LeducEnv:
    def __init__(self, seed: int | None = None) -> None:
        self._raw = rlcard.make("leduc-holdem")      # 2‑player limit game
        if seed is not None:
            self._raw.seed(seed)
        self._current_player: int
        self.reset()

    # ------------------------------------------------------------------ #
    # Public helpers                                                     #
    # ------------------------------------------------------------------ #

    def reset(self) -> Tuple[Any, List[int]]:
        """Start a new hand; return (obs, legal_actions) for player 0."""
        state, self._current_player = self._raw.reset()
        return self._format(state)

    def observe(self) -> Tuple[Any, List[int]]:
        """Peek at the state for the current player."""
        return self._format(self._raw.get_state(self._current_player))

    def step(self, action: int) -> Tuple[Tuple[Any, List[int]], List[int], bool]:
        """
        Execute `action` for the *current* player and return:

            (next_obs, next_legals), reward_vector, done

        * `next_obs`, `next_legals` are **None / []** after a terminal step.
        * `reward_vector` is length‑2, with non‑zeros only when `done` is True.
        """
        next_state, next_player = self._raw.step(action)

        if self._raw.is_over():                        # terminal node
            reward = self._raw.get_payoffs()           # length‑2 list
            done = True
            next_obs, next_legals = None, []
        else:                                          # game continues
            reward = [0, 0]
            done = False
            self._current_player = next_player
            next_obs, next_legals = self._format(next_state)

        return (next_obs, next_legals), reward, done

    # ------------------------------------------------------------------ #
    # Internal helper                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format(state: dict) -> Tuple[Any, List[int]]:
        """
        Compress RLCard’s verbose state dict into the tuple we expose:

            obs         –NumPy array of features
            legal       –list of int action codes (0 fold | 1 call | 2 raise | 3 check)
        """
        obs   = state["obs"]
        legal = list(state["legal_actions"].keys())
        return obs, legal

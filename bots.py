"""Two deterministic benchmark opponents used for sanity‑tests."""

import random
from typing import List, Any

# RLCard’s Leduc action‑encoding:
#   0 = fold   1 = call   2 = raise
FOLD, CALL, RAISE = 0, 1, 2

class AlwaysCallAgent:
    def step(self, obs: Any, legal: List[int]) -> int:
        return CALL if CALL in legal else random.choice(legal)

class AlwaysFoldAgent:
    def step(self, obs: Any, legal: List[int]) -> int:
        return FOLD if FOLD in legal else random.choice(legal)

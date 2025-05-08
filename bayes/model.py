"""
Bayesian opponent‑style model
------------------------------------
Network structure  (DAG):

Exactly the textbook definition of a belief network, each node
is conditionally independent of all non‑parents given its parents
"""

from __future__ import annotations
from enum import IntEnum, auto
import numpy as np
import pandas as pd
from typing import Dict, List

# ---------- discrete variables -----------------------------------------

class Style(IntEnum):
    TAG = 0   # tight‑aggressive
    LAG = auto()
    TP  = auto()   # tight‑passive
    LP  = auto()

class Action(IntEnum):
    FOLD  = 0
    CALL  = 1
    RAISE = 2

# ---------- helper -----------------------------------------------------

def _dirichlet_map(rows: int, cols: int, alpha: float = 1.0) -> np.ndarray:
    """alpha‑symmetric Dirichlet prior counts."""
    return np.full((rows, cols), alpha, dtype=np.float64)

# ---------- main class -------------------------------------------------

class StyleBN:
    """
    Minimal hand‑rolled Bayesian network with Dirichlet‑multinomial
    learning (conjugate prior to closed‑form posterior).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        # P(Style)  1x4   prior counts
        self.prior = np.full(len(Style), alpha)

        # CPTs:  Style Action   (4x3)  per street
        self.cpt_pf  = _dirichlet_map(len(Style), len(Action), alpha)
        self.cpt_fl  = _dirichlet_map(len(Style), len(Action), alpha)

    # ---------- observation update -------------------------------------

    def update(self, pf_action: Action, fl_action: Action) -> None:
        """
        One complete hand gives exactly two observations (pre‑flop +
        flop).  We treat Style as latent and integrate it out with
        Bayes rule as in textbook 12.3.2
        """
        # P(Style | evidence so far)  (length‑4)
        post = self.posterior_style()

        # increment Dirichlet counts ‑ expectation step
        self.prior += post
        self.cpt_pf[np.arange(len(Style)), pf_action] += post
        self.cpt_fl[np.arange(len(Style)), fl_action] += post

    # ---------- queries -------------------------------------------------

    def posterior_style(self) -> np.ndarray:
        """Return current P(Style) after marginalising CPTs."""
        return self.prior / self.prior.sum()

    def predict(self, street: str) -> Dict[Action, float]:
        """
        Return predictive distribution over villain’s next action on the
        given street (\"preflop\" | \"flop\")  via
        Sum over style  P(A | style) P(style)   – textbook chain rule
        """
        if street == "preflop":
            cpt = self.cpt_pf
        elif street == "flop":
            cpt = self.cpt_fl
        else:
            raise ValueError("street must be 'preflop' or 'flop'")

        style_prob = self.posterior_style()
        action_prob = (cpt / cpt.sum(1, keepdims=True)).T @ style_prob
        return {Action(i): p for i, p in enumerate(action_prob)}

    # ---------- pretty print -------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            self.cpt_pf / self.cpt_pf.sum(1, keepdims=True),
            index=[s.name for s in Style],
            columns=[a.name for a in Action],
        )
        df["P(Style)"] = self.posterior_style()
        return df

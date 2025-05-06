"""
Tiny helpers that turn raw RLCard hand histories into the categorical
evidence our BN expects.
"""

from __future__ import annotations
from typing import List
from .model import Action

def _action_from_rlcard(code: int) -> Action:
    # RLCard: 0 fold, 1 call, 2 raise/check (we map check to call)
    if code == 0:
        return Action.FOLD
    if code == 1 or code == 3:
        return Action.CALL
    return Action.RAISE

def vpip_pfr_from_history(history: List[int]) -> tuple[bool, bool]:
    """
    Given a *single hand* history (list of RLCard action codes issued by
    villain pre‑flop) return
        VPIP  (voluntarily put $ in pot)  –bool
        PFR   (pre‑flop raise)           –bool
    Those bits feed the symbolic rule base later.
    """
    vpip = any(a in (1, 2, 3) for a in history)   # call or raise
    pfr  = any(a == 2 for a in history)           # raise only
    return vpip, pfr

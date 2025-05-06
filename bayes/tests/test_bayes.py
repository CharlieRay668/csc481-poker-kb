from bayes import StyleBN, Action, Style
import numpy as np

def test_posterior_shift():
    bn = StyleBN(alpha=1.0)
    # Simulate villain who open‑raises 9/10 hands
    for _ in range(20):
        bn.update(Action.RAISE, Action.CALL)

    probs = bn.posterior_style()
    assert probs[Style.LAG] >= probs.mean() # LAG should be more likely
    pred = bn.predict("preflop")
    assert pred[Action.RAISE] > 0.5

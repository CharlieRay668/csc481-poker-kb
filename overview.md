
# Technical Overview

## 1  Environment layer

### `leduc_env.py`

| Symbol | Role | Notes |
|--------|------|-------|
| `class LeducEnv` | Thin wrapper over **RLCard**’s `"leduc-holdem"` | Exposes only `reset()`, `observe()`, `step()`. Hides RLCard’s verbose state dict. |
| `_format(state)` | Internal helper | Converts RLCard’s nested dict to `(obs, legal)` tuple so downstream code stays library‑agnostic. |
| `self._current_player` | Tracked manually | RLCard keeps it internally; we surface it for simpler hand loops. |

Why not extend `gym.Env`? – We will later, but keeping zero dependencies here helps early CI.

---

### `bots.py`

| Class | Behaviour | Use‑case |
|-------|-----------|----------|
| `AlwaysCallAgent` | Tries `CALL`; if illegal (edge cases) falls back to random legal action | Baseline for unit smoke‑tests |
| `AlwaysFoldAgent` | Tries `FOLD` first | Ensures fold path is exercised |

---

## 2  Symbolic KB layer (`kb/`)

### `ontology.md`

*Human‑readable* vocabulary table. Categories follow KRR textbook Chapter 3.

* **Sorts** – basic object domains (`card/1`, `suit/1` …)  
* **Fluents** – time‑varying facts (e.g., `pot_size/1`)  
* **Static predicates** – never change within a hand (`better_pos/2`)  
* **Derived predicates** – proved by rules (`strong_pair/1`)  
* **Action constants** – the agent’s motor vocabulary.

### `poker_rules.pl`

*SWI‑Prolog module* implementing first strategic rules.

| Pred | Arity | Meaning |
|------|-------|---------|
| `rank_value/2` | (Rank,Int) | Hard‑codes King > Queen > Jack for Leduc. |
| `strong_pair/1` | Hand | True when hand contains a pair ≥ Queens. |
| `good_open/1` | Hand | Union of `strong_pair` and King‑Queen offsuit scramble. |
| `best_action/4` | Street,Hand,Board,Advice | Main decision predicate (early stub). |
| `cbet_ok/2` | Hand,Board | Flop continuation‑bet heuristic. |
| `villain_style/2` | VPIP,StyleAtom | Maps raw VPIP ratio to tight/loose atom; will be invoked by production rules later. |

Coding style: pure *definite* Horn clauses → guarantees linear‑time SLD resolution.

### `rules.json`

Forward‑chaining mirror of critical rules expressed as production rules:

* Fields: `name`, `if` (array of pattern strings), `then` (array of WM actions).  
* Variables prefixed with `?` for later binding in RETE.

This duplication allows us to **swap reasoning engines** without rewriting content.

---

## 3  Probabilistic layer (`bayes/`)

### `model.py`

| Symbol | Kind | Description |
|--------|------|-------------|
| `Style` (Enum) | {TAG,LAG,TP,LP} | Four canonical play‑styles. |
| `Action` (Enum) | {FOLD,CALL,RAISE} | Discrete action variable across streets. |
| `class StyleBN` | Main Bayesian network | Maintains Dirichlet counts for: **P(Style)** and two CPTs: Style → Action\_(preflop|flop). |
| `__init__(alpha)` | ctor | Initializes symmetric Dirichlet prior α on every multinomial. |
| `update(pf_action, fl_action)` | Online learning | Implements Dirichlet‑multinomial conjugate update per hand (no MCMC needed). |
| `posterior_style()` | Query | Returns 4‑vector of posterior probabilities over Style. |
| `predict(street)` | Query | Predictive distribution over villain’s next action using law of total probability Σ\_style. |
| `to_frame()` | Convenience | Pandas DataFrame view for real‑time logging/plots. |

Design decisions:

* **No external BN libs** → easier grading, zero dep headaches.  
* **Latent Style integrated analytically** (textbook §12.3.2) → O(K) per update where K=4.

### `features.py`

Utility module turning raw RLCard action codes into categorical evidence.

* `_action_from_rlcard(code)` – maps RLCard int to `Action` enum.  
* `vpip_pfr_from_history(history)` – returns `(vpip_bool, pfr_bool)` for Milestone 4’s symbolic engine.

### `__init__.py`

Re‑exports `StyleBN`, `Action`, `Style`, and `vpip_pfr_from_history` for ergonomic imports.

### `tests/test_bayes.py`

Regression test ensuring:

1. After 20 mostly‑raise hands, `P(LAG)` overtakes the prior mean.  
2. Pre‑flop `RAISE` prediction exceeds 0.5.

---

## 4  Unit tests (`tests/` and `bayes/tests/`)

* `test_leduc_env.py` – plays 1 000 hands Always‑Call vs Always‑Fold to validate the wrapper never crashes.  
* `test_bayes.py` – sanity‑check Bayesian drift.

---
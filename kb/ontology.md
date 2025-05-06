# Poker‑KB Ontology — pass #1
This ontology follows the knowledge engineering checklist in Ch. 3

| **Category** | **Symbol** | **English role** | **Example literal** |
|--------------|------------|------------------|---------------------|
| **Sorts (types)** | `card/1`, `rank/1`, `suit/1`, `street/1`, `player/1`, `style/1`, `pos/1` | Atomic objects | `rank(king)` |
| **Fluents (F‑)*** | `pot_size(N)`, `current_street(S)`, `stack(Player,N)`, `bet_to_call(N)` | Time‑varying facts | `current_street(flop)` |
| **Static predicates (P‑)** | `better_pos(A,B)` (“A is in position on B”), `has_pair(Player)` | Snapshot properties | `has_pair(hero)` |
| **Derived predicates (D‑)** | `strong_pair(Hand)`, `preflop_premium(Hand)`, `draw_heavy(Board)` | Computed via rules | `strong_pair([k,h,k,d])` |
| **Action constants** | `fold`, `call`, `raise`, `check`, `bet` | Action symbols | `action(raise,hero)` |

\*We treat pot size, stacks … as simple facts, not functions, because RLCard’s state parser returns scalar values each decision tick.

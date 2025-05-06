:- begin_tests(kb).

:- use_module('../kb/poker_rules').

test(open_raise) :-
        best_action(preflop, [king,king,2], 0, raise).

test(cbet) :-
        best_action(flop, [queen,7], [queen,9,2], bet).

:- end_tests(kb).

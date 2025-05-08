/*  Knowledge‑base    :   Poker Leduc */

:- module(poker_rules, [best_action/4, villain_style/2]).

%% --------------------------------------------------------------------
%%  Domain‑level helper rules
%% --------------------------------------------------------------------

rank_value(king, 3).
rank_value(queen,2).
rank_value(jack, 1).

% A hand contains a pair if any rank occurs at least twice
pair(Hand) :-
    select(R, Hand, Rest),
    member(R, Rest).

% A strong pair is a pair of rank >= 2 (queen or king)
strong_pair(Hand) :-
    pair(Hand),
    rank_value(R, V), V >= 2,
    include(==(R), Hand, Matches),
    length(Matches, N),
    N >= 2.


top_pair(Board, Hand) :-
        Board = [Com|_],
        member(Com, Hand),
        rank_value(Com, V), V >= 2.

%% --------------------------------------------------------------------
%%  small strategic knowledge  (enough to test the pipeline)
%% --------------------------------------------------------------------

/* Pre‑flop opening decisions */
good_open(Hand)         :- strong_pair(Hand).
good_open(Hand)         :- Hand = [king, queen | _].

cbet_ok(Hand, Board)    :- top_pair(Board, Hand).
cbet_ok(_Hand, Board)  :- draw_heavy(Board).

best_action(preflop, Hand, _Pot, raise) :-
        good_open(Hand), !.
best_action(preflop, _Hand, _Pot, call).

/* Flop continuation‑bet heuristic */
best_action(flop, Hand, Board, bet) :-
        cbet_ok(Hand, Board), !.

/* Simple villain style tagging */
vpip_thresh(loose,    0.35).
vpip_thresh(tight,    0.25).

villain_style(Vpip, loose) :-
        vpip_thresh(loose,T), Vpip  >= T.
villain_style(Vpip, tight) :-
        vpip_thresh(tight,T), Vpip  <  T.

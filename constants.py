# This file defines the core constants for the Leduc Hold'em game.

# Card ranks used in the game
RANKS = ['J', 'Q', 'K']

# The deck consists of two of each rank
DECK = ['J', 'J', 'Q', 'Q', 'K', 'K']

# Numerical values assigned to each rank for comparison
RANK_VALUE = {'J': 0, 'Q': 1, 'K': 2}

# Maximum number of raises allowed in the pre-flop betting round
MAX_RAISES_PREFLOP = 2

# Maximum number of raises allowed in the post-flop betting round
MAX_RAISES_POSTFLOP = 2

# Initial number of chips each player contributes to the pot (ante)
ANTE = 1

# Bet/Raise size for the pre-flop round
PREFLOP_BET_SIZE = 2

# Bet/Raise size for the post-flop round (typically double the pre-flop)
POSTFLOP_BET_SIZE = 2
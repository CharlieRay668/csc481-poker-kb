# Leduc Hold'em Adaptive Bayesian-Counterfactual Regret Minimization Bot: CSC 481 Final Project

This is a final project for Cal Poly's CSC 481: Knowledge Based System course, taught by Dr. Rodrigo Canaan.
This project was completed by Charlie Ray and Ethan Ellis.

## What is this project?

This project implements an adaptive Leduc Hold'em poker bot, with the idea of tyring to exploit players who are playing
sub-optimally, better than just the nash strategy.

We achieve this by creating a Bayesian opponent model, where we observe previous hand history and build an estimated
opponent strategy distribution. Then we run CFR against this estimated opponent strategy to generate a best response policy.

The hypothesis is that we will be able to generate a strategy that has a higher average chip per hand return than
a strategy that just plays the Nash Equilibrium strategy.

## How to run this project:

All requirements are listed in the requirements.txt file. Simply run
```
pip install -r requirements.txt
```
to install all dependancies.

After that, you should be able to just run
```
python3 simulate.py
```
to generate all the simulation plots found in our report.

## Sample simulation results.

Because the simulation takes a while to run, we have included sample simulation results below.
If you choose to run the simulation, all charts should be stored in the "results" directory. You can also
run clean_images.py to combine plots, which will be saved in the "cleaned_results" directory.

### Aggressive Opponents
![Aggressive Opponents](https://github.com/CharlieRay668/csc481-poker-kb/blob/master/cleaned_results/aggressive_opponents_combined.png)

### Loose Opponents
![Loose Opponents](https://github.com/CharlieRay668/csc481-poker-kb/blob/master/cleaned_results/loose_opponents_combined.png)

### Passive Opponents
![Passive Opponents](https://github.com/CharlieRay668/csc481-poker-kb/blob/master/cleaned_results/passive_opponents_combined.png)

### Special Opponents
![Special Opponents](https://github.com/CharlieRay668/csc481-poker-kb/blob/master/cleaned_results/special_opponents_combined.png)

### Tight Opponents
![Tight Opponents](https://github.com/CharlieRay668/csc481-poker-kb/blob/master/cleaned_results/tight_opponents_combined.png)


# Sovereign-MOE
Cutting-plane generation for VYUHA's from-scratch MILP solver. Implements Gomory mixed-integer cuts from fractional simplex tableau rows, and knapsack cover cuts via greedy separation over binary constraints, with numerical safeguards rejecting cuts of poor fractionality, high coefficient dynamism, or weak efficacy, keeping the root bound tight.

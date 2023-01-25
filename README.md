# Environment-sensitive-generalization-and-exploration
Repository of the research project 'Environment-sensitive generalization and exploration strategies'

## Simulate decision-making
* simulation.py simulates the decision-making behaviour of one computational agent with a randomly sampled strategy
For full parameter study; call this function e.g. 1000 times.
* solving_models.py contains the functions to solve a grid
* show_grid.py can be use for grid visualization

## Estimate model parameters
* parameter_estimation.py code: For models with only one environment (and three model parameters to estimate), use the estimate_1env function, for models with two environment (and 8 models for (non)-environment-sensitive strategies), use the estimate_2env function.
* parameter_estimation_behavioural.py  reads in the behavioural data of one particiant.

## Parameter recovery
* parameter_recovery.py to recover the 1 env model, includes the generating of the behavioural data of one computational agent (with a randomly sampled strategy)
For full recovory study; call this function e.g. 1000 times.

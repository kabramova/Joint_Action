"""
This is the main file for running evolution of neural network agents in the Knoblich and Jordan (2003) task.
"""
import random
from evolve import Evolution
import numpy as np

# evolution parameters
POP_SIZE = 30  # population size
MAX_GENS = 101  # maximum generations to evolve: make sure this is one more than a multiple of check_int
MUTATION_VAR = 1  # mutation variance
PROB_CROSSOVER = 0.8  # crossover probability
ELITIST_FRAC = 0.1  # fraction of population to copy without modification to the new generation
FPS_FRAC = 0.7  # fraction of population that reproduces (the rest is filled with new random agents)
CHECK_INT = 10  # interval (in generations) of how often to save the current population to file

# network parameters
NUM_NEURONS = 8  # number of neurons in a network
STEP_SIZE = 0.01  # Euler step size and simulation step size TODO: should they be the same?
TAU_RANGE = (1, 10)
THETA_RANGE = (-15, 15)
W_RANGE = (-15, 15)
G_RANGE = (1, 1)

# evaluation parameters
# VELOCITIES = [3.3, 4.3, -3.3, -4.3]  # target velocities
# IMPACT = [0.7, 1.0]  # tracker's button press impact
# SCREEN_WIDTH = [-20,20]  # width of the environment

VELOCITIES = [4, -4]  # target velocities
IMPACT = [1.0]  # tracker's button press impact
SCREEN_WIDTH = [-20,20]  # width of the environment
START_PERIOD = 100


def main():
    # set random seed
    random.seed(592)

    evolution_params = [MAX_GENS, MUTATION_VAR, PROB_CROSSOVER, ELITIST_FRAC, FPS_FRAC, CHECK_INT]
    network_params = [NUM_NEURONS, STEP_SIZE, TAU_RANGE, THETA_RANGE, W_RANGE, G_RANGE]
    evaluation_params = [SCREEN_WIDTH, VELOCITIES, IMPACT, START_PERIOD]

    # set up evolution
    evolution = Evolution(POP_SIZE, evolution_params, network_params, evaluation_params)

    # run evolution
    evolution.run()

    return


if __name__ == '__main__':
    main()

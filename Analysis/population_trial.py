import numpy as np
import simulate
import evolve


def main():
    # evolution parameters
    pop_size = 10
    max_gens = 10  # maximum generations to evolve
    mutation_var = 1
    prob_crossover = 0.8
    elitist_frac = 0.1
    fps_frac = 0.7
    check_int = 10  # interval (in generations) of how often to dump the current search state

    # network parameters
    n_neurons = 8
    step_size = 0.01
    tau_range = (1, 10)
    theta_range = (-15, 15)
    w_range = (-15, 15)
    g_range = (1, 1)

    # evaluation parameters
    velocities = [3.3, 4.3, -3.3, -4.3]
    impact = [0.7, 1.0]
    screen_width = [-20, 20]

    evolution_params = [max_gens, mutation_var, prob_crossover, elitist_frac, fps_frac, check_int]
    network_params = [n_neurons, step_size, tau_range, theta_range, w_range, g_range]
    evaluation_params = [velocities, impact]

    evolution = evolve.Evolution(pop_size, evolution_params, network_params, evaluation_params)
    population = evolution.create_population(pop_size)

    for agent in population:
        simulation_run = simulate.Simulation(screen_width, step_size, evaluation_params[0], evaluation_params[1])
        trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)  # returns a list of fitness in all trials
        agent.fitness = np.mean(trial_data['fitness'])


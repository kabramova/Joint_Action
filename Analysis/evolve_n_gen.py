import numpy as np
import simulate
import evolve
import pickle
import matplotlib.pyplot as plt

def main():
    # evolution parameters
    pop_size = 50
    max_gens = 21  # maximum generations to evolve
    mutation_var = 1
    prob_crossover = 0.8
    elitist_frac = 0.1
    fps_frac = 0.8
    check_int = 5  # interval (in generations) of how often to dump the current search state

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
    evaluation_params = [screen_width, velocities, impact]

    evolution = evolve.Evolution(pop_size, evolution_params, network_params, evaluation_params)
    population = evolution.create_population(pop_size)

    avgfit = []
    bestfit = []

    gen = 0
    while gen < max_gens:
        print('Generation {}'.format(gen))
        for agent in population:
            simulation_run = simulate.Simulation(evaluation_params[0], step_size, evaluation_params[1], evaluation_params[2])
            trial_data = simulation_run.run_trials(agent, simulation_run.trials)  # returns a list of fitness in all trials
            agent.fitness = np.mean(trial_data['fitness'])

        # log fitness results
        population_avg_fitness = np.mean([agent.fitness for agent in population])
        avgfit.append(round(population_avg_fitness, 3))
        bestfit.append(round(max([agent.fitness for agent in population]), 3))
        # print("Average fitness in generation {} is {}".format(gen, round(population_avg_fitness, 3)))

        new_population = evolution.reproduce(population)

        # save the intermediate population
        if gen % check_int == 0:
            popfile = open('./Agents/gen{}'.format(gen), 'wb')
            pickle.dump(population, popfile)
            popfile.close()

        population = new_population
        gen += 1

    fits = [avgfit, bestfit]
    fitfile = open('./Agents/fitnesses', 'wb')
    pickle.dump(fits, fitfile)
    fitfile.close()

    # plt.plot(avgfit)
    # plt.plot(bestfit)
    # plt.show()

if __name__ == '__main__':
    main()
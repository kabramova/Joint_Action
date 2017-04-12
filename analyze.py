import pickle
import simulate
import matplotlib.pyplot as plt
import json


def run_single_trial(generation_num, agent_num, trial_num):
    json_data = open('config.json')
    config = json.load(json_data)
    json_data.close()

    pop_file = open('./Agents/gen{}'.format(generation_num), 'rb')
    population = pickle.load(pop_file)
    pop_file.close()
    population.sort(key=lambda agent: agent.fitness, reverse=True)
    agent = population[agent_num]

    simulation_run = simulate.Simulation(config['network_params']['step_size'], config['evaluation_params'])

    trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)
    # trial_data.keys()
    # ['keypress', 'target_pos', 'input-output', 'tracker_pos', 'brain_state', 'tracker_v'])

    # plot results
    plt.plot(trial_data['target_pos'][trial_num], label='Target position')
    plt.plot(trial_data['tracker_pos'][trial_num], label='Tracker position')
    plt.plot(trial_data['tracker_v'][trial_num], label='Tracker velocity')
    plt.plot(trial_data['keypress'][trial_num][:, 0], label='Left keypress')
    plt.plot(trial_data['keypress'][trial_num][:, 1], label='Right keypress')
    plt.legend()
    plt.title("Best agent, generation {}".format(generation_num))
    plt.show()
    return trial_data


td = run_single_trial(150, 0, 0)


def plot_fitness():

    fit_file = open('./Agents/fitnesses', 'rb')
    fits = pickle.load(fit_file)
    fit_file.close()

    plt.plot(fits[0], label="Average population fitness")
    plt.plot(fits[1], label="Best agent fitness")
    plt.legend()
    plt.title("Fitness over generations")
    plt.show()

plot_fitness()

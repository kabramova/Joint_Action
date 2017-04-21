import pickle
import simulate
import matplotlib.pyplot as plt
import json
from evolve import Evolution


def run_single_trial(generation_num, agent_num):
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

    num_trials = len(trial_data['target_pos'])

    # plot results
    fig = plt.figure(figsize=(10,6))
    fig.suptitle("Best agent, generation {}".format(generation_num))

    for p in range(num_trials):
        ax = fig.add_subplot(2, 2, p+1)
        ax.plot(trial_data['target_pos'][p], label='Target position')
        ax.plot(trial_data['tracker_pos'][p], label='Tracker position')
        ax.plot(trial_data['tracker_v'][p], label='Tracker velocity')
        ax.plot(trial_data['keypress'][p][:, 0], label='Left keypress')
        ax.plot(trial_data['keypress'][p][:, 1], label='Right keypress')
    plt.legend()
    plt.show()

    return trial_data


td = run_single_trial(250, 0)


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


def run_random_agent():
    # load configuration settings
    json_data = open('config.json')
    config = json.load(json_data)
    json_data.close()

    evolution = Evolution(config['evolution_params']['pop_size'],
                          config['evolution_params'],
                          config['network_params'],
                          config['evaluation_params'],
                          config['agent_params'])
    population = evolution.create_population(1)
    agent = population[0]

    simulation_run = simulate.Simulation(config['network_params']['step_size'], config['evaluation_params'])

    trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)
    # trial_data.keys()
    # ['keypress', 'target_pos', 'input-output', 'tracker_pos', 'brain_state', 'tracker_v'])

    num_trials = len(trial_data['target_pos'])

    # plot results
    fig = plt.figure(figsize=(10,6))
    fig.suptitle('Random agent')

    for p in range(num_trials):
        ax = fig.add_subplot(2, 2, p+1)
        ax.plot(trial_data['target_pos'][p], label='Target position')
        ax.plot(trial_data['tracker_pos'][p], label='Tracker position')
        ax.plot(trial_data['tracker_v'][p], label='Tracker velocity')
        ax.plot(trial_data['keypress'][p][:, 0], label='Left keypress')
        ax.plot(trial_data['keypress'][p][:, 1], label='Right keypress')
    plt.legend()
    plt.show()

    return trial_data

td2 = run_random_agent()

def run_random_population(size):
    json_data = open('config.json')
    config = json.load(json_data)
    json_data.close()

    evolution = Evolution(config['evolution_params']['pop_size'],
                          config['evolution_params'],
                          config['network_params'],
                          config['evaluation_params'],
                          config['agent_params'])
    population = evolution.create_population(size)
    for agent in population:
        simulation_run = simulate.Simulation(evolution.step_size, evolution.evaluation_params)
        trial_data = simulation_run.run_trials(agent, simulation_run.trials)  # returns a list of fitness in all trials
        # agent.fitness = np.mean(trial_data['fitness'])
        # agent.fitness = evolution.harmonic_mean(trial_data['fitness'])
        agent.fitness = min(trial_data['fitness'])
    population.sort(key=lambda agent: agent.fitness, reverse=True)
    agent = population[0]
    print(agent.fitness)
    simulation_run = simulate.Simulation(evolution.step_size, evolution.evaluation_params)
    trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)  # returns a list of fitness in all trials

    num_trials = len(trial_data['target_pos'])

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle('Random agent')

    for p in range(num_trials):
        ax = fig.add_subplot(2, 2, p+1)
        ax.plot(trial_data['target_pos'][p], label='Target position')
        ax.plot(trial_data['tracker_pos'][p], label='Tracker position')
        ax.plot(trial_data['tracker_v'][p], label='Tracker velocity')
        ax.plot(trial_data['keypress'][p][:, 0], label='Left keypress')
        ax.plot(trial_data['keypress'][p][:, 1], label='Right keypress')
    plt.legend()
    plt.show()


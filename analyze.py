import pickle
import simulate
import matplotlib.pyplot as plt


def run_single_trial(generation_num, agent_num, trial_num, velocities, impact, startperiod):
    step_size = 0.01
    screen_width = [-20, 20]

    popfile = open('./Agents/gen{}'.format(generation_num), 'rb')
    population = pickle.load(popfile)
    popfile.close()
    population.sort(key=lambda agent: agent.fitness, reverse=True)
    agent = population[agent_num]

    simulation_run = simulate.Simulation(screen_width, step_size, velocities, impact, startperiod)
    trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)
    # trial_data.keys()
    # ['keypress', 'target_pos', 'input-output', 'tracker_pos', 'brain_state', 'tracker_v'])

    # plot results
    plt.plot(trial_data['target_pos'][trial_num])
    plt.plot(trial_data['tracker_pos'][trial_num])
    plt.plot(trial_data['tracker_v'][trial_num])
    plt.plot(trial_data['keypress'][trial_num])
    plt.show()
    return trial_data


velocities = [4, -4]
impact = [1]
startperiod = 100

td = run_single_trial(300, 0, 0, velocities, impact, startperiod)


def plot_fitness():

    fitfile = open('./Agents/fitnesses', 'rb')
    fits = pickle.load(fitfile)
    fitfile.close()

    plt.plot(fits[0])
    plt.plot(fits[1])
    plt.show()

plot_fitness()
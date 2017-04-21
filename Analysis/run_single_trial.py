import sys
sys.path.append("..")

import CTRNN
import simulate
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle


def main(mode, generation):
    step_size = 0.01
    # evaluation parameters
    screen_width = [-20, 20]
    velocities = [3.3, 4.3, -3.3, -4.3]
    impact = [0.7, 1.0]
    #velocities = [3.3]
    #impact = [1.0]

    if mode == 'random':
        # network parameters
        n_neurons = 8
        tau_range = (1, 10)
        theta_range = (-15, 15)
        w_range = (-15, 15)
        g_range = (1, 1)

        agent_brain = CTRNN.CTRNN(n_neurons, step_size, tau_range, g_range, theta_range, w_range)
        agent = simulate.Agent(agent_brain)

    else:
        popfile = open('./Agents/gen{}'.format(generation), 'rb')
        population = pickle.load(popfile)
        popfile.close()

        population.sort(key=lambda agent: agent.fitness, reverse=True)
        agent = population[0]

    simulation_run = simulate.Simulation(screen_width, step_size, velocities, impact)
    trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)
    # trial_data.keys()
    # ['keypress', 'target_pos', 'input-output', 'tracker_pos', 'brain_state', 'tracker_v'])
    trial_num = 0

    # plot results
    plt.plot(trial_data['target_pos'][trial_num])
    plt.plot(trial_data['tracker_pos'][trial_num])
    plt.plot(trial_data['tracker_v'][trial_num])
    plt.plot(trial_data['keypress'][trial_num])
    plt.show()


# # calculate and make sure all works fine
# tau = agent.brain.Tau
# theta = agent.brain.Theta
# g = agent.brain.G
# w = agent.brain.W
#
# tgp1 = trial_data['target_pos'][0]
# trp1 = trial_data['tracker_pos'][0]
# trv1 = trial_data['tracker_v'][0]
# io1 = trial_data['input-output'][0]
# br1 = trial_data['brain_state'][0]
# kp1 = trial_data['keypress'][0]
#
# tgp2 = trial_data['target_pos'][1]
# trp2 = trial_data['tracker_pos'][1]
# trv2 = trial_data['tracker_v'][1]
# io2 = trial_data['input-output'][1]
# br2 = trial_data['brain_state'][1]
# kp2 = trial_data['keypress'][1]
#
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# o = sigmoid(np.multiply(g, br1 + theta))
# np.hstack((agent.VW, agent.AW, agent.MW))
# input = np.zeros(n_neurons)
# input[7] = io1[0] * trp1
# input[1] = io1[1] * tgp1
# input[0] = np.sum([io1[2] * tgp1, io1[3] * trp1])
#
# dy_dt = np.multiply(1 / tau, - br1 + np.dot(w, o) + input) * step_size
# y = br1 + dy_dt
#
# n4out = br1[3]
# n6out = br1[5]
#
# activation_left = np.sum([n4out * io1[8], n6out * io1[10]])
# activation_right = np.sum([n4out * io1[9], n6out * io1[11]])
#
#
#
# # measure time taken
# start_time = time.time()
# trial_data2 = simulation_run.run_trials(agent, simulation_run.trials)
# elapsed_time = time.time() - start_time

# Modeling the stepwise adjustment of a participant's link distance by means of left- and right-clicks was tricky,
# because it required mapping the CTRNN neuron outputs from continuous dynamics to a discrete domain.
# We chose to model a mouse click by implementing a button activation threshold. If a button neuron's output (range [0, 1])
# increases to more than or equal to 0.75, then its button is turned “on” and produces a “click.”
# The button is turned “off” when that neuron's output falls below 0.75. In this way an agent cannot adjust
# its link continuously, because the button has to be turned off before it can be turned back on. The reason
# for these choices is to facilitate a distinction between the timescales of movement and link adjustment,
# which should be faster and slower, respectively. We modeled the activities of the two buttons with
# two distinct neurons, rather than with two activation thresholds of one neuron, because we believed that
# this might facilitate the evolution of flexible behavior.


if __name__ == '__main__':
    main('saved', 10)
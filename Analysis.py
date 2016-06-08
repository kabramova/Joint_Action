import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Info: http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

from SA_Evolution import *
from JA_Evolution import *


'''
TODO:
    - Behaviour compare with paper
    - analysis of strategy.
    - analysis William Beer 2015 (p.8)
    - Attractor space
    - Statistic for meaningful difference between sound off/on (run evolution n-times, for x-Generations)

GRAPHS:
1) GRAPH A:
 - Position Target and Tracker (y-axis), time (x-axis)
'''
# Setup Agent(s) to analyse:
condition = single_or_joint_request()
audicon = audio_condition_request()

load = load_request()

if load is False:
    filename = filename_request(condition)  # "joint" or "single"
    # filename = "Gen1001-2000.popsize55.mut0.02.sound_cond=False.JA.joint(Fitness6.1)"

    if condition == "single":
        sa = SA_Evolution(auditory_condition=audicon)
        if isinstance(filename, str):
           sa_performance = sa.reimplement_population(filename=filename, Plot=True)
           sa_performance = np.array(sa_performance, dtype=object)
           fitness = np.round(sa.pop_list[0, 1],2)
           np.save("./Analysis/single/sa_performance_cond{}_fitness{}".format(sa.condition, fitness), sa_performance)
           # sa_performance[0-3] are the different trials
           # sa_performance[0-3][0-5] = fitness[0], trajectories[1], keypress[2], sounds[3], neural_state[4], neural_input_L[5]

    if condition == "joint":
        ja = JA_Evolution(auditory_condition=audicon, pop_size=55)
        if isinstance(filename, str):
            ja_performance = ja.reimplement_population(filename=filename, Plot=True)
            ja_performance = np.array(ja_performance, dtype=object)
            fitness = np.round(ja.pop_list_L[0,1],2)
            np.save("./Analysis/joint/ja_performance_cond{}_fitness{}".format(ja.condition, fitness), ja_performance)
            # ja_performance[0-3] are the different trials
            # ja_performance[0-3][0-7] = fitness[0], trajectories[1], keypress[2], sounds[3], neural_state_L[4], neural_state_L[5], neural_input_L[6], neural_input_L[7]


# len(sa_performance)
# sa_performance[1][0] # Fitness for particular speed+direction trial
if load is True:
    if condition == "single":
        sa_performance = load_file(condition, audicon)
        print(">> File is loaded in sa_performance")
        fitness = np.round(np.mean([i[0] for i in sa_performance]),2) # Fitness over all trials
    if condition == "joint":
        ja_performance = load_file(condition, audicon)
        print(">> File is loaded in ja_performance")
        fitness = np.round(np.mean([i[0] for i in ja_performance]),2) # Fitness over all trials


## Split in different Trials:
sl = sa_performance[0] if condition == "single" else ja_performance[0] # speed: slow, initial target-direction: left
sr = sa_performance[1] if condition == "single" else ja_performance[1] # speed: slow, initial target-direction: right
fl = sa_performance[2] if condition == "single" else ja_performance[2] # speed: fast, initial target-direction: left
fr = sa_performance[3] if condition == "single" else ja_performance[3] # speed: fast, initial target-direction: right

trials = [sl, sr, fl, fr]
trial_names = ["slowleft", "slowright", "fastleft", "fastright"]
index = -1

folder = "./Analysis/graphs/{}_{}_{}".format(condition, audicon, fitness)
if not os.path.exists(folder):
    os.mkdir(folder)

# TODO: adapt graphs to joint condition (if needed)
for trial in trials:
    index += 1
    trial_name = trial_names[index]

    ## GRAPH A:
    tracker = trial[1][:,0] # trajectories[1], tracker: tracs[:,0]
    target  = trial[1][:,1] # trajectories[1], target:  tracs[:,1]

    fig_a = plt.figure("GRAPH A, Trial {}".format(trial_name))
    plt.plot(tracker,'r', markersize=12, alpha=0.5)
    plt.plot(target, 'g')
    plt.savefig("./{}/{} GRAPH A (POSITIONS) Trial {}  [WiP]".format(folder, condition, trial_name))
    plt.close(fig_a)

    ## GRAPH B:
    # neural_state[4]
    trial[4].shape  # knoblin.Y
    neural_state = trial[4]

    # neural_input_L[5]
    trial[5].shape  # knoblin.I
    neural_input = trial[5]


    # Info: http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    fig_b = plt.figure("GRAPH B, Trial {}".format(trial_name))
    ax = fig_b.add_subplot(111, projection='3d')
    for i in range(neural_state.shape[1]):
        ax.plot(xs = range(neural_state.shape[0]), zs = neural_state[:,i], ys=np.repeat(i+1,neural_state.shape[0]))
        ax.plot(xs = range(neural_input.shape[0]), zs = neural_input[:,i], ys=np.repeat(i+1,neural_state.shape[0]), alpha=0.0)
    # plt.plot(neural_input, alpha=0.3)
    plt.savefig("./{}/{} GRAPH B (Neural Activity) Trial {}  [WiP]".format(folder, condition, trial_name))
    plt.close(fig_b)


    fig_b_b = plt.figure("GRAPH B_b, Trial {}".format(trial_name))
    ax = fig_b_b.add_subplot(111, projection='3d')
    for i in range(neural_state.shape[1]):
        ax.plot(xs = range(neural_state.shape[0]), zs = neural_state[:,i], ys=np.repeat(i+1,neural_state.shape[0]), alpha=0.3)
        ax.plot(xs = range(neural_input.shape[0]), zs = neural_input[:,i], ys=np.repeat(i+1,neural_state.shape[0]))
    plt.savefig("./{}/{} GRAPH B_b (Neural Activity) Trial {}  [WiP]".format(folder, condition, trial_name))
    plt.close(fig_b_b)


    # TODO: Maybe Wireframe:
    # fig_b_c = plt.figure("GRAPH B, Trial {}".format(trial_name))
    # ax = fig_b_c.add_subplot(111, projection='3d')
    # for i in range(neural_state.shape[1]):
    #     ax.plot_wireframe(X = range(neural_state.shape[0]), Z = neural_state[:,i], Y=i+1)
    # # plt.plot(neural_input, alpha=0.3)
    # plt.savefig("./{}/{} GRAPH B_C (Neural Activity) WIRE Trial {}  [WiP]".format(folder, condition, trial_name))
    # plt.close(fig_b_c)


    # TODO: Contour plots
    # fig_b_d = plt.figure("GRAPH B_b, Trial {}".format(trial_name))
    # ax = fig_b_d.add_subplot(111, projection='3d')
    # for i in range(neural_state.shape[1]):
    #     ax.counter(X = range(neural_state.shape[0]), Z = neural_state[:,i], Y=i+1, alpha=0.3)
    #     ax.counter(X = range(neural_input.shape[0]), Z = neural_input[:,i], Y=i+1)
    # plt.savefig("./graphs/{} GRAPH B_b (Neural Activity) Trial {}  [WiP]".format(folder, condition, trial_name))
    # plt.close(fig_b_d)


    ## GRAPH C:
    # keypress[2], sounds[3]
    fig_c = plt.figure("GRAPH C, Trial {}".format(trial_name))
    plt.ylim(-1.1, 1.1)

    for i in range(len(trial[2])):
        if trial[2][i,0] == 1: # keypress left
            plt.plot(i, trial[2][i,0]+1, 'gs', markersize=8)

        if trial[2][i, 1] == 1:  # keypress right
            plt.plot(i, trial[2][i, 0]-1, 'gs', markersize=8)

    for i in range(len(trial[3])):
        if trial[3][i,0] == -1: # sound left
            plt.plot(i, trial[3][i,0]+1, 'yo', markersize=12, alpha=0.3)

        if trial[3][i, 1] == 1:  # sound right
            plt.plot(i, trial[3][i, 0]-1, 'yo', markersize=12, alpha=0.3)

    plt.savefig("./{}/{} GRAPH C (Keypress and Sound) Trial {}  [WiP]".format(folder, condition, trial_name))
    plt.close(fig_c)


    ## GRAPH D:
    # keypress[2]
    # tracker = trial[1][:,0] # trajectories[1], tracker: tracs[:,0]
    # target  = trial[1][:,1] # trajectories[1], target:  tracs[:,1]

    # for negative change of target position (left movement)
    fig_d_left = plt.figure("GRAPH D left, Trial {}".format(trial_name))

    # Define boarders
    plt.xlim(-20,1, 20,1)
    plt.ylim(-20,1, 20,1)

    for row in range(len(trial[2])):
        #TODO: Here "L" and "R" plotten, instead of dots
        if trial[2][row, 0] == -1:     # left
            if target[row] < target[row-1]:  # check whether left movement
                plt.plot(target[row], tracker[row], "bo")
        if trial[2][row, 1] == 1:      # right
            if target[row] < target[row - 1]:
                plt.plot(target[row], tracker[row], "ro")

            plt.savefig("./{}/{} GRAPH D Left (Keypress and Trajectories) Trial {}  [WiP]".format(folder, condition, trial_name))

    # for positive change of target position (right movement)
    fig_d_right = plt.figure("GRAPH D right, Trial {}".format(trial_name))

    # Define boarders
    plt.xlim(-20,1, 20,1)
    plt.ylim(-20,1, 20,1)

    for row in range(len(trial[2])):
        #TODO: Here "L" and "R" plotten, instead of dots
        if trial[2][row, 0] == -1:     # left
            if target[row] > target[row-1]:  # check whether right movement
                plt.plot(target[row], tracker[row], "bo")
        if trial[2][row, 1] == 1:      # right
            if target[row] > target[row - 1]:
                plt.plot(target[row], tracker[row], "ro")

    plt.savefig("./{}/{} GRAPH D Right (Keypress and Trajectories) Trial {}  [WiP]".format(folder, condition, trial_name))
    plt.close(fig_d_left)
    plt.close(fig_d_right)


    ## GRAPH E:
    # # neural_state[4]
    # trial[4].shape  # knoblin.Y
    # neural_state = trial[4]
    #
    # # neural_input_L[5]
    # trial[5].shape  # knoblin.I
    # neural_input = trial[5]


    # Plot Input-receptor Neuron 1, and out motor-neurons 4 & 6
    fig_e = plt.figure("GRAPH E, Trial {}".format(trial_name))
    ax = fig_e.add_subplot(111, projection='3d')
    ax.plot(xs = neural_state[:,0], zs = neural_state[:,3], ys=neural_state[:,5])
    #TODO: name axes!

    plt.savefig("./{}/{} GRAPH E (Neural Activity of Neuron 1,4,6) Trial {}  [WiP]".format(folder,
                                                                                           condition,
                                                                                           trial_name))
    plt.close(fig_e)


    # Plot average Neural-state and Trajectories (Target, Tracker)
    fig_e_b = plt.figure("GRAPH E, Trial {}".format(trial_name))
    ax = fig_e_b.add_subplot(111, projection='3d')
    # TODO: name axes! x,y should be from range_boarders (-20,20), colour it (red/blue) the higher/lower it gets

    average = [np.mean(i) for i in neural_state]
    ax.plot(xs=target,
            ys=tracker,
            zs=average)


    plt.savefig("./{}/{} GRAPH E_b (Average Neural Activity and trajectories) Trial {}  [WiP]".format(folder,
                                                                                                      condition,
                                                                                                      trial_name))
    plt.close(fig_e_b)
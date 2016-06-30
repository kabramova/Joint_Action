import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D # Info: http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
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

filename = filename_request(condition)  # "joint" or "single"
# filename = "Gen1001-2000.popsize55.mut0.02.sound_cond=False.JA.joint(Fitness6.1)"

load = load_request()

lesion = lesion_request() # True/False
lesion_name = "_lesion" if lesion else ""

if load is False:

    if condition == "single":
        sa = SA_Evolution(auditory_condition=audicon)
        if isinstance(filename, str):
            sa_performance = sa.reimplement_population(filename=filename, plot=True, lesion=lesion)
            sa_performance = np.array(sa_performance, dtype=object)
            sa.implement_genome(genome_string=sa.pop_list[0, 2:])
            fitness = np.round(np.mean([i[0] for i in sa_performance]), 2)  # Fitness over all trials
            # fitness = np.round(sa.pop_list[0, 1], 2)
            np.save("./Analysis/single/sa_performance_cond{}_fitness{}{}".format(sa.condition, fitness, lesion_name), sa_performance)
            # sa_performance[0-3] are the different trials
            # sa_performance[0-3][0-5] = fitness[0], trajectories[1], keypress[2], # sounds[3], neural_state[4],
            # neural_input_L[5]

    if condition == "joint":
        ja = JA_Evolution(auditory_condition=audicon, pop_size=55)
        if isinstance(filename, str):
            ja_performance = ja.reimplement_population(filename=filename, plot=True, lesion=lesion)
            ja_performance = np.array(ja_performance, dtype=object)
            ja.implement_genome(genome_string=ja.pop_list_L[0, 2:], side="left")
            ja.implement_genome(genome_string=ja.pop_list_R[0, 2:], side="right")
            fitness = np.round(np.mean([i[0] for i in ja_performance]), 2)  # Fitness over all trials
            # fitness = np.round(ja.pop_list_L[0, 1], 2)
            np.save("./Analysis/joint/ja_performance_cond{}_fitness{}{}".format(ja.condition, fitness, lesion_name), ja_performance)
            # ja_performance[0-3] are the different trials
            # ja_performance[0-3][0-7] = fitness[0], trajectories[1], keypress[2], sounds[3], neural_state_L[4],
            # neural_state_R[5], neural_input_R[6], neural_input_L[7]


# len(sa_performance)
# sa_performance[1][0] # Fitness for particular speed+direction trial
if load is True:
    if condition == "single":
        sa_performance = load_file(condition, audicon)
        print(">> File is loaded in sa_performance")
        fitness = np.round(np.mean([i[0] for i in sa_performance]), 2)  # Fitness over all trials

        sa = SA_Evolution(auditory_condition=audicon)
        if isinstance(filename, str):
            sa.reimplement_population(filename=filename, plot=False)
            sa.implement_genome(genome_string=sa.pop_list[0,2:])
            # sa_performance[0-3] are the different trials
            # sa_performance[0-3][0-5] = fitness[0], trajectories[1], keypress[2], # sounds[3], neural_state[4],
            # neural_input_L[5]

    if condition == "joint":
        ja_performance = load_file(condition, audicon)
        print(">> File is loaded in ja_performance")
        fitness = np.round(np.mean([i[0] for i in ja_performance]), 2)  # Fitness over all trials

        ja = JA_Evolution(auditory_condition=audicon, pop_size=55)
        if isinstance(filename, str):
            ja.reimplement_population(filename=filename, plot=False)
            ja.implement_genome(genome_string=ja.pop_list_L[0, 2:], side="left")
            ja.implement_genome(genome_string=ja.pop_list_R[0, 2:], side="right")

# Split in different Trials:
sl = sa_performance[0] if condition == "single" else ja_performance[0]  # speed: slow, initial target-direction: left
sr = sa_performance[1] if condition == "single" else ja_performance[1]  # speed: slow, initial target-direction: right
fl = sa_performance[2] if condition == "single" else ja_performance[2]  # speed: fast, initial target-direction: left
fr = sa_performance[3] if condition == "single" else ja_performance[3]  # speed: fast, initial target-direction: right

trials = [sl, sr, fl, fr]
trial_names = ["slowleft", "slowright", "fastleft", "fastright"]
index = -1

folder = "./Analysis/graphs/{}_{}_{}{}".format(condition, audicon, fitness, lesion_name)
if not os.path.exists(folder):
    os.mkdir(folder)

# Colours:
# http://matplotlib.org/examples/color/colormaps_reference.html
# cmap = plt.get_cmap("Paired")

col = ["royalblue", "tomato", "palegreen", "fuchsia", "gold", "darkviolet", "darkslategray", "orange"]  # colors.cnames

# for i in range(8):
#     plt.plot(2*i,1, marker="o", c=col[i])
#     plt.xlim(-1,15)

# trial = trials[0]

# TODO: Save neural states in csv (choose specific trial, or all):
# if condition == "single":
#   np.savetxt("{}/neural_state_fastleft.csv".format(folder), sa_performance[2][4], delimiter=";")

# if condition == "joint":
#   np.savetxt("{}/neural_state_L_slowleft.csv".format(folder), ja_performance[0][4], delimiter=";")
#   np.savetxt("{}/neural_state_R_slowleft.csv".format(folder), ja_performance[0][5], delimiter=";")

for trial in trials:

    index += 1
    trial_name = trial_names[index]

    # Create Folder
    current_folder = "{}/{}".format(folder, trial_name)
    if not os.path.exists(current_folder):
        os.mkdir(current_folder)

    tracker = trial[1][:, 0]  # trajectories[1], tracker: tracs[:,0]
    target = trial[1][:, 1]   # trajectories[1], target:  tracs[:,1]

    # Define Regions (2 Border-, 1 Middle Region):
    upper_bound = int(np.round(max(target)))
    lower_bound = int(np.round(min(target)))
    screen_width = upper_bound-lower_bound
    region_width = screen_width/3
    right_border = 0 + region_width/2
    left_border = 0 - region_width/2
    simlength = len(tracker)

    # Find time points, when tracker enters new region:
    crossing = [cross > right_border or cross < left_border for cross in target]
    crosses = []
    test = True
    for i, bol in enumerate(crossing):
        if bol == test:
            crosses.append(i)
            test = not test

    # GRAPH A:
    fig_a = plt.figure("GRAPH A, Trial {}".format(trial_name), figsize=(10, 2), dpi=80)
    plt.xlim(0-10.0, simlength+10.0)
    plt.ylim(-20.5, 20.5)
    # Plot Regions
    plt.plot(range(0, simlength), np.repeat(right_border, simlength), "--", c="grey", alpha=0.2)
    plt.plot(range(0, simlength), np.repeat(left_border, simlength), "--", c="grey", alpha=0.2)
    for cross in crosses:
        plt.plot(np.repeat(cross, len(range(lower_bound, upper_bound))), range(lower_bound, upper_bound), "--", c="grey", alpha=0.2)

    plt.plot(tracker, 'r', markersize=12, alpha=0.5, label="Tracker")
    plt.plot(target, 'g', label="Target")

    plt.legend()
    # plt.title("Target and Tracker Positions")
    plt.xlabel("Timesteps")
    plt.ylabel("Position")

    plt.savefig("./{}/{} GRAPH A (POSITIONS) Trial {}".format(current_folder, condition, trial_name))
    plt.close(fig_a)

    # GRAPH B:
    # Single: neural_state[4]
    # Single: neural_input_L[5]
    # trial[4].shape
    # trial[5].shape
    if condition == "single":
        neural_state = trial[4]  # knoblin.Y
        neural_input = trial[5]  # knoblin.I

    # Joint:  neural_state_L[4], neural_state_R[5]
    # Joint:  neural_input_L[6], neural_input_R[7]
    if condition == "joint":
        neural_state_L = trial[4]
        neural_state_R = trial[5]
        neural_input_L = trial[6]
        neural_input_R = trial[7]

    # Info: http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    fig_b = plt.figure("GRAPH B, Trial {}".format(trial_name))
    ax = fig_b.add_subplot(111, projection='3d')
    if condition == "single":
        for i in range(neural_state.shape[1]):
            ax.plot(xs=range(neural_state.shape[0]), zs=neural_state[:, i], ys=np.repeat(i+1, neural_state.shape[0]))
            ax.plot(xs=range(neural_input.shape[0]), zs=neural_input[:, i], ys=np.repeat(i+1, neural_state.shape[0]),
                    alpha=0.0)

    if condition == "joint":
        for i in range(neural_state_L.shape[1]):
            ax.plot(xs=range(len(neural_state_L)), zs=neural_state_L[:, i], ys=np.repeat(i + 1, len(neural_state_L)),
                    alpha=.5,
                    c=col[i]) # c=cmap(i**3))
            ax.plot(xs=range(len(neural_state_R)), zs=neural_state_R[:, i], ys=np.repeat(i + 1, len(neural_state_R)),
                    c=col[i]) # c=cmap(i**3))

            ax.plot(xs=range(len(neural_state_L)), zs=neural_input_L[:, i], ys=np.repeat(i + 1, len(neural_state_L)),
                    alpha=0.0)
            ax.plot(xs=range(len(neural_state_R)), zs=neural_input_R[:, i], ys=np.repeat(i + 1, len(neural_state_R)),
                    alpha=0.0)

    # ax.set_title("Neural activation through trial")
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Neurons')
    ax.set_zlabel('Activation')
    plt.savefig("./{}/{} GRAPH B (Neural Activity) Trial {}".format(current_folder, condition, trial_name))
    plt.close(fig_b)

    # Leave input neurons out:
    fig_b_a = plt.figure("GRAPH B_a, Trial {}".format(trial_name))
    ax = fig_b_a.add_subplot(111, projection='3d')
    if condition == "single":
        for i in [2, 3, 4, 5, 6]:
            ax.plot(xs=range(neural_state.shape[0]), zs=neural_state[:, i], ys=np.repeat(i + 1, neural_state.shape[0]))
            ax.plot(xs=range(neural_input.shape[0]), zs=neural_input[:, i], ys=np.repeat(i + 1, neural_state.shape[0]),
                    alpha=0.0)

    if condition == "joint":
        for i in [2, 3, 4, 5, 6]:
            ax.plot(xs=range(len(neural_state_L)), zs=neural_state_L[:, i], ys=np.repeat(i + 1, len(neural_state_L)),
                    alpha=.5,
                    c=col[i])  # c=cmap(i**3))
            ax.plot(xs=range(len(neural_state_R)), zs=neural_state_R[:, i], ys=np.repeat(i + 1, len(neural_state_R)),
                    c=col[i])  # c=cmap(i**3))

            ax.plot(xs=range(len(neural_state_L)), zs=neural_input_L[:, i], ys=np.repeat(i + 1, len(neural_state_L)),
                    alpha=0.0)
            ax.plot(xs=range(len(neural_state_R)), zs=neural_input_R[:, i], ys=np.repeat(i + 1, len(neural_state_R)),
                    alpha=0.0)

    # ax.set_title("Neural activation through trial")
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Neurons')
    ax.set_zlabel('Activation')
    plt.savefig("./{}/{} GRAPH B_a (Neural Activity) without Input Neurons Trial {}".format(current_folder, condition, trial_name))
    plt.close(fig_b_a)

    # Input
    fig_b_b = plt.figure("GRAPH B_b, Trial {}".format(trial_name))
    ax = fig_b_b.add_subplot(111, projection='3d')

    if condition == "joint":
        for i in range(neural_input_L.shape[1]):
            '''
            ax.plot(xs = range(neural_state.shape[0]), zs = neural_state[:,i], ys=np.repeat(i+1,neural_state.shape[0]),
                alpha=0.1)
            '''
            ax.plot(xs=range(len(neural_input_L)), zs=neural_input_L[:, i], ys=np.repeat(i+1, len(neural_input_L)),
                    alpha=.5,
                    c=col[i])  # c=cmap(i**3))

            ax.plot(xs=range(len(neural_input_R)), zs=neural_input_R[:, i], ys=np.repeat(i + 1, len(neural_input_R)),
                    c=col[i])  # c=cmap(i**3))

    if condition == "single":
        for i in range(neural_input.shape[1]):
            # ax.plot(xs = range(neural_state.shape[0]), zs = neural_state[:,i],
            # ys=np.repeat(i+1, neural_state.shape[0]), alpha=0.1)
            ax.plot(xs=range(len(neural_input)), zs=neural_input[:, i], ys=np.repeat(i + 1, len(neural_input)),
                    # ls="-.",
                    c=col[i])  # c=cmap(i**3))

    ax.set_title("Neural Input")
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Neurons')
    ax.set_zlabel('weighted Input')

    plt.savefig("./{}/{} GRAPH B_b (Neural Activity) Trial {}".format(current_folder, condition, trial_name))
    plt.close(fig_b_b)

    # GRAPH C:
    # keypress[2], sounds[3]
    fig_c = plt.figure("GRAPH C, Trial {}".format(trial_name), figsize=(10, 2), dpi=80)
    plt.xlim(0, len(trial[2]))
    plt.ylim(2, -2)
    plt.xlabel("Timesteps")
    plt.ylabel("Keypress")
    plt.yticks([-1, 1], ["left", "right"])

    # Plot verticals, when target enters new region
    for cross in crosses:
        plt.plot(np.repeat(cross, len(range(lower_bound, upper_bound))), range(lower_bound, upper_bound), "--", c="grey", alpha=0.2)

    for i in range(len(trial[3])):
        if trial[3][i, 0] == 1:  # sound left
            plt.plot(i, trial[3][i, 0]-2, 'yo', markersize=16, alpha=0.05, lw=0)

        if trial[3][i, 1] == 1:  # sound right
            plt.plot(i, trial[3][i, 1], 'yo', markersize=16, alpha=0.05, lw=0)

    for i in range(len(trial[2])):
        if trial[2][i, 0] == -1:  # keypress left
            plt.plot(i, trial[2][i, 0], 'bs', markersize=8)

        if trial[2][i, 1] == 1:  # keypress right
            plt.plot(i, trial[2][i, 1], 'bs', markersize=8)

    plt.savefig("./{}/{} GRAPH C (Keypress and Sound) Trial {}".format(current_folder, condition, trial_name))
    plt.close(fig_c)

    # GRAPH D:
    # keypress[2]
    # tracker = trial[1][:,0] # trajectories[1], tracker: tracs[:,0]
    # target  = trial[1][:,1] # trajectories[1], target:  tracs[:,1]

    # for negative change of target position (left movement)

    fig_d_neg = plt.figure("GRAPH D delta-, Trial {}".format(trial_name), figsize=(6, 6), dpi=80)
    # Define borders
    plt.xlim(-20.5, 20.5)
    plt.ylim(-20.5, 20.5)

    # Label Axes, Title
    plt.title("$\delta- position \ of \ Target$")
    plt.xlabel("Position Target")
    plt.ylabel("Position Tracker")

    # Plot lines for borderregion
    plt.plot(range(lower_bound, upper_bound), np.repeat(left_border, len(range(lower_bound, upper_bound))), "--", c="grey", alpha=0.2)
    plt.plot(range(lower_bound, upper_bound), np.repeat(right_border, len(range(lower_bound, upper_bound))), "--", c="grey", alpha=0.2)
    plt.plot(np.repeat(left_border, len(range(lower_bound, upper_bound))), range(lower_bound, upper_bound), "--", c="grey", alpha=0.2)
    plt.plot(np.repeat(right_border, len(range(lower_bound, upper_bound))), range(lower_bound, upper_bound),  "--", c="grey", alpha=0.2)

    # Plot
    for row in range(len(trial[2])):
        if target[row] < target[row-1]:  # check whether left movement
            if row % 20 == 0:
                plt.plot(target[row], tracker[row], marker="o", alpha=.4, lw=0, c="blue", ms=.4)

            if trial[2][row, 0] == -1:     # left
                    plt.plot(target[row], tracker[row], marker=r"$ {} $".format("L"), markersize=10,
                             markerfacecolor="blue")
            if trial[2][row, 1] == 1:      # right
                    plt.plot(target[row], tracker[row], marker=r"$ {} $".format("R"), ms=10, mfc="red")

    plt.savefig("./{}/{} GRAPH D delta- (Keypress and Trajectories) Trial {}".format(current_folder,
                                                                                     condition,
                                                                                     trial_name))

    # for positive change of target position (right movement)
    fig_d_pos = plt.figure("GRAPH D delta+, Trial {}".format(trial_name), figsize=(6, 6), dpi=80)

    # Define borders
    plt.xlim(-20.5, 20.5)
    plt.ylim(-20.5, 20.5)

    # Label Axes, Title
    plt.title("$\delta+ position \ of \ Target$")
    plt.xlabel("Position Target")
    plt.ylabel("Position Tracker")

    # Plot lines for borderregion
    plt.plot(range(lower_bound, upper_bound), np.repeat(left_border, len(range(lower_bound, upper_bound))), "--", c="grey", alpha=0.2)
    plt.plot(range(lower_bound, upper_bound), np.repeat(right_border, len(range(lower_bound, upper_bound))), "--", c="grey", alpha=0.2)
    plt.plot(np.repeat(left_border, len(range(lower_bound, upper_bound))), range(lower_bound, upper_bound), "--", c="grey", alpha=0.2)
    plt.plot(np.repeat(right_border, len(range(lower_bound, upper_bound))), range(lower_bound, upper_bound), "--", c="grey", alpha=0.2)

    # Plot
    for row in range(len(trial[2])):
        if target[row] > target[row - 1]:  # check whether right movement
            if row % 20 == 0:
                plt.plot(target[row], tracker[row], marker="o", alpha=.4, lw=0, c="blue", ms=.4)

            if trial[2][row, 0] == -1:     # left
                    plt.plot(target[row], tracker[row], marker=r"$ {} $".format("L"), markersize=10,
                             markerfacecolor="blue")
            if trial[2][row, 1] == 1:      # right
                    plt.plot(target[row], tracker[row], marker=r"$ {} $".format("R"), ms=10, mfc="red")

    plt.savefig("./{}/{} GRAPH D delta+ (Keypress and Trajectories) Trial {}".format(current_folder,
                                                                                     condition,
                                                                                     trial_name))

    plt.close(fig_d_neg)
    plt.close(fig_d_pos)

    # GRAPH E:
    # # neural_state[4]
    # trial[4].shape  # knoblin.Y
    # neural_state = trial[4]
    #
    # # neural_input_L[5]
    # trial[5].shape  # knoblin.I
    # neural_input = trial[5]

    # Plot Input-receptor Neuron 1, and output motor-neurons 4 & 6
    fig_e = plt.figure("GRAPH E, Trial {}".format(trial_name))
    ax = fig_e.add_subplot(111, projection='3d')

    # Label Axes, title
    ax.set_title("States of Input-receptor Neuron 1, and output motor-neurons 4 & 6")
    ax.set_xlabel('Neuron 4')
    ax.set_ylabel('Neuron 6')
    ax.set_zlabel('Neuron 1')

    # Plot
    if condition == "single":
        ax.plot(xs=neural_state[:, 3], ys=neural_state[:, 5], zs=neural_state[:, 0], color="red")

    if condition == "joint":
        ax.plot(xs=neural_state_L[:, 3], ys=neural_state_L[:, 5], zs=neural_state_L[:, 0],
                color="red", label="Left Agent")
        ax.plot(xs=neural_state_R[:, 3], ys=neural_state_R[:, 5], zs=neural_state_R[:, 0],
                color="blue", label="Right Agent")
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., fancybox=True)

    plt.savefig("./{}/{} GRAPH E (Neural Activity of Neuron 1,4,6) Trial {}".format(current_folder,
                                                                                    condition,
                                                                                    trial_name))
    plt.close(fig_e)

    # Plot average Neural-state and Trajectories (Target, Tracker)
    if condition == "single":
        average = [np.mean(i) for i in neural_state]

    if condition == "joint":
        average_L = [np.mean(l) for l in neural_state_L]
        average_R = [np.mean(r) for r in neural_state_R]

    fig_e_b = plt.figure("GRAPH E, Trial {}".format(trial_name))
    ax = fig_e_b.add_subplot(111, projection='3d')

    # axes limits
    ax.set_xlim(-20.5, 20.5)
    ax.set_ylim(-20.5, 20.5)

    # Label Axes, title
    ax.set_title('Network excitation')
    ax.set_xlabel('Target Position')
    ax.set_ylabel('Tracker Position')
    ax.set_zlabel('Average Neural State')

    # set color
    colorsMap = 'jet'
    cm = plt.get_cmap(colorsMap)
    if condition == "single":
        cNorm = matplotlib.colors.Normalize(vmin=min(average), vmax=max(average))
    else:  # condition == "joint":
        cNorm = matplotlib.colors.Normalize(vmin=min(min(average_L), min(average_R)),
                                            vmax=max(max(average_L), max(average_R)))

    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    if condition == "single":
        ax.scatter(xs=target, ys=tracker, zs=average, c=scalarMap.to_rgba(average), lw=0, s=1.5)

    if condition == "joint":
        # c=scalarMap.to_rgba(average_L):
        ax.scatter(xs=target, ys=tracker, zs=average_L, c="red", lw=0, s=1.5, label="Left Agent")
        # c=scalarMap.to_rgba(average_R):
        ax.scatter(xs=target, ys=tracker, zs=average_R, c="blue", lw=0, s=1.5, label="Right Agent")
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.,
                  fancybox=True, markerscale=3)

    # scalarMap.set_array(average)
    # fig_e_b.colorbar(scalarMap)

    plt.savefig("./{}/{} GRAPH E_b (Network excitation and trajectories) Trial {}".format(current_folder,
                                                                                          condition,
                                                                                          trial_name))
    plt.close(fig_e_b)

    # GRAPH F:
    fig_f = plt.figure("GRAPH F, Motor Neurons, Target-Tracker Distance Trial {}".format(trial_name))
    ax = fig_f.add_subplot(111, projection='3d')

    # axes limits
    ax.set_zlim(-20.5, 20.5)

    # Label Axes, title
    ax.set_xlabel('Neuron 4')
    ax.set_ylabel('Neuron 6')
    ax.set_zlabel('Distance Target-Tracker')

    # target-tracker: Distance
    distance = target-tracker

    # Plot
    if condition == "single":
        ax.plot(xs=neural_state[:, 3], ys=neural_state[:, 5], zs=distance, color="darkviolet")

        for row in range(len(trial[2])):
            if trial[2][row, 0] == -1:  # left press
                ax.scatter(neural_state[row, 3], neural_state[row, 5], zs=distance[row], marker=r"$ {} $".format("L"),
                           s=30, lw=0, c="blue")
            if trial[2][row, 1] == 1:   # right
                plt.scatter(neural_state[row, 3], neural_state[row, 5], zs=distance[row], marker=r"$ {} $".format("R"),
                            s=30, lw=0, c="red")

    if condition == "joint":
        ax.plot(xs=neural_state_L[:, 3], ys=neural_state_L[:, 5], zs=distance,
                color="royalblue", label="Left Agent")
        ax.plot(xs=neural_state_R[:, 3], ys=neural_state_R[:, 5], zs=distance,
                color="fuchsia", label="Right Agent")

        for row in range(len(trial[2])):
            if trial[2][row, 0] == -1:  # left press
                ax.scatter(neural_state_L[row, 3], neural_state_L[row, 5], zs=distance[row],
                           marker=r"$ {} $".format("L"),
                           s=30, lw=0, c="blue")
            if trial[2][row, 1] == 1:  # right
                plt.scatter(neural_state_R[row, 3], neural_state_R[row, 5], zs=distance[row],
                            marker=r"$ {} $".format("R"), s=30, lw=0, c="red")

        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., fancybox=True)

    plt.savefig("./{}/{} GRAPH F (Motor Neuron Activity of Neuron 4, 6 and Distance Target-Tracker) Trial {}".format(current_folder,
                                                                                                                     condition,
                                                                                                                     trial_name))

    plt.close(fig_f)

    # GRAPH G:
    # TODO: Dynamical Graph (Neural state y, df/dy)

    # Y = []
    # for i in range(sa.simlength*2):
    #     Y.append(sa.knoblin.Y)
    #     sa.knoblin.next_state()
    #
    # meanY = [np.mean(i) for i in Y]
    #
    # for i in range(sa.simlength*2):
    #     for j in range(sa.knoblin.Y.shape[0]):
    #         plt.plot(i,Y[i][j], marker="o", markeredgewidth=0.0, ms=1)
    #
    # plt.plot(meanY)

    # DYDT = []
    # Y = np.matrix(np.zeros((len(sa.knoblin.Y),1)))
    # for i in np.arange(-20,21):
    #     tempY = np.matrix(np.zeros((len(sa.knoblin.Y),1)))
    #     tempY[0] = i
    #     Y = np.append(Y,tempY,1)
    #
    # for i in range(Y.shape[1]):
    #     O = sigmoid(np.multiply(sa.knoblin.G, Y[:,i] + sa.knoblin.Theta))
    #     DYDT.append(np.multiply(1 / sa.knoblin.Tau, - Y[:,i] + np.dot(sa.knoblin.W, O) + sa.knoblin.I) * sa.knoblin.h)
    #
    #
    # for i in range(len(DYDT)):
    #     for j in range(len(sa.knoblin.Y)):
    #         plt.plot(i, DYDT[i][j], marker="o", ms=5., markeredgewidth=0.0, c=col[j])

    # TODO: plot weights
    # http: // matplotlib.org / examples / specialty_plots / hinton_demo.html


def _blob(x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    plt.fill(xcorners, ycorners, colour, edgecolor=colour)


def hinton(W, maxweight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    Temporarily disables matplotlib interactive mode if it is on,
    otherwise this takes forever.
    """
    reenable = False
    if plt.isinteractive():
        plt.ioff()
        reenable = True

    plt.clf()
    height, width = W.shape
    if not maxweight:
        maxweight = 2 ** np.ceil(np.log(np.max(np.abs(W))) / np.log(2))

    plt.fill(np.array([0, width, width, 0]),
             np.array([0, 0, height, height]),
             'gray')

    plt.axis('off')
    plt.axis('equal')
    for x in range(width):
        for y in range(height):
            _x = x + 1
            _y = y + 1
            w = W[y, x]
            if w > 0:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, w / maxweight),
                      'white')
            elif w < 0:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, -w / maxweight),
                      'black')
    if reenable:
        plt.ion()


if condition == "single":

    # Weights
    hinton(sa.knoblin.W, maxweight=sa.knoblin.W_RANGE[1])
    plt.title("Knoblin Hinton diagram - Weights 8x8")
    plt.show()
    plt.savefig("./{}/{} GRAPH W, Knoblin Weights".format(folder, condition, trial_name))
    plt.close()

    if audicon:
        # Audio Weights
        hinton(sa.knoblin.WA.transpose(), maxweight=sa.knoblin.W_RANGE[1])
        plt.title("Knoblin Hinton diagram - Audio Weights 4x1")
        plt.show()
        plt.savefig("./{}/{} GRAPH WA, Knoblin Audio Weights".format(folder, condition))
        plt.close()

    # Vision Weights
    hinton(sa.knoblin.WV.transpose(), maxweight=sa.knoblin.W_RANGE[1])
    plt.title("Knoblin Hinton diagram - Vision Weights 4x1")
    plt.show()
    plt.savefig("./{}/{} GRAPH WV, Knoblin Vision Weights".format(folder, condition))
    plt.close()

    # Motor Weights
    hinton(sa.knoblin.WM.transpose(), maxweight=sa.knoblin.W_RANGE[1])
    plt.title("Knoblin Hinton diagram - Motor Weights 4x1")
    plt.show()
    plt.savefig("./{}/{} GRAPH WM, Knoblin Motor Weights".format(folder, condition))
    plt.close()

if condition == "joint":

    # Weights
    hinton(ja.knoblin_L.W, maxweight=ja.knoblin_L.W_RANGE[1])
    plt.title("Left Knoblin Hinton diagram - Weights 8x8")
    plt.show()
    plt.savefig("./{}/{} GRAPH W, Left Knoblin Weights".format(folder, condition))
    plt.close()

    hinton(ja.knoblin_R.W, maxweight=ja.knoblin_R.W_RANGE[1])
    plt.title("Right Knoblin Hinton diagram - Weights 8x8")
    plt.show()
    plt.savefig("./{}/{} GRAPH W, Right Knoblin Weights".format(folder, condition))
    plt.close()

    if audicon:
        # Audio Weights
        hinton(ja.knoblin_L.WA.transpose(), maxweight=ja.knoblin_L.W_RANGE[1])
        plt.title("Left Knoblin Hinton diagram - Audio Weights 4x1")
        plt.show()
        plt.savefig("./{}/{} GRAPH WA, Left Knoblin Audio Weights".format(folder, condition))
        plt.close()

        hinton(ja.knoblin_R.WA.transpose(), maxweight=ja.knoblin_R.W_RANGE[1])
        plt.title("Right Knoblin Hinton diagram - Audio Weights 4x1")
        plt.show()
        plt.savefig("./{}/{} GRAPH WA, Right Knoblin Audio Weights".format(folder, condition))
        plt.close()

    # Vision Weights
    hinton(ja.knoblin_L.WV.transpose(), maxweight=ja.knoblin_L.W_RANGE[1])
    plt.title("Left Knoblin Hinton diagram - Vision Weights 4x1")
    plt.show()
    plt.savefig("./{}/{} GRAPH WV, Left Knoblin Vision Weights".format(folder, condition))
    plt.close()

    hinton(ja.knoblin_R.WV.transpose(), maxweight=ja.knoblin_R.W_RANGE[1])
    plt.title("Right Knoblin Hinton diagram - Vision Weights 4x1")
    plt.show()
    plt.savefig("./{}/{} GRAPH WV, Right Knoblin Vision Weights".format(folder, condition))
    plt.close()

    # Motor Weights
    hinton(ja.knoblin_L.WM.transpose(), maxweight=ja.knoblin_L.W_RANGE[1])
    plt.title("Left Knoblin Hinton diagram - Motor Weights 4x1")
    plt.show()
    plt.savefig("./{}/{} GRAPH WM, Left Knoblin Motor Weights".format(folder, condition))
    plt.close()

    hinton(ja.knoblin_R.WM.transpose(), maxweight=ja.knoblin_R.W_RANGE[1])
    plt.title("Right Knoblin Hinton diagram - Motor Weights 4x1")
    plt.show()
    plt.savefig("./{}/{} GRAPH WM, Right Knoblin Motor Weights".format(folder, condition))
    plt.close()


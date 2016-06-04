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
number_of_generations = 2
filename = filename_request(condition)  # "joint" or "single"
# filename = "Gen1001-2000.popsize55.mut0.02.sound_cond=False.JA.joint(Fitness6.1)"

# TODO: check here, whether external file to this agents is available.

if condition == "single":
    sa = SA_Evolution(auditory_condition=audicon)
    if isinstance(filename, str):
       sa_performance = sa.reimplement_population(filename=filename, Plot=True)
       # sa_performance[0-3] are the different trials
       # sa_performance[0-3][0-5] = fitness[0], trajectories[1], keypress[2], sounds[3], neural_state[4], neural_input_L[5]

if condition == "joint":
    ja = JA_Evolution(auditory_condition=audicon, pop_size=55)
    if isinstance(filename, str):
        ja_performance = ja.reimplement_population(filename=filename, Plot=True)
        # ja_performance[0-3] are the different trials
        # ja_performance[0-3][0-7] = fitness[0], trajectories[1], keypress[2], sounds[3], neural_state_L[4], neural_state_L[5], neural_input_L[6], neural_input_L[7]



len(sa_performance)
sa_performance[1][0] # Fitness for particular speed+direction trial

# TODO: save .._performance in external file.

## Split in different Trials:
sl = sa_performance[0] # speed: slow, initial target-direction: left
sr = sa_performance[1] # speed: slow, initial target-direction: right
fl = sa_performance[2] # speed: fast, initial target-direction: left
fr = sa_performance[3] # speed: fast, initial target-direction: right

## GRAPH A:
tracker = sr[1][:,0] # trajectories[1], tracker: tracs[:,0]
target  = sr[1][:,1] # trajectories[1], target:  tracs[:,1]

fig_a = plt.figure("GRAPH A")
plt.plot(tracker,'r', markersize=12, alpha=0.5)
plt.plot(target, 'g')
plt.savefig("./graphs/GRAPH A (POSITIONS) [WiP]")
plt.close(fig_a)

## GRAPH B:
# neural_state[4]
sr[4].shape  # knoblin.Y
neural_state = sr[4]

# neural_input_L[5]
sr[5].shape  # knoblin.I
neural_input = sr[5]


# Info: http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
fig_b = plt.figure("GRAPH B")
ax = fig_b.add_subplot(111, projection='3d')
for i in range(neural_state.shape[1]):
    ax.plot(xs = range(neural_state.shape[0]), zs = neural_state[:,i], ys=np.repeat(i+1,neural_state.shape[0]))
    ax.plot(xs = range(neural_input.shape[0]), zs = neural_input[:,i], ys=np.repeat(i+1,neural_state.shape[0]), alpha=0.0)
# plt.plot(neural_input, alpha=0.3)
plt.savefig("./graphs/GRAPH B (Neural Activity) [WiP]")
plt.close(fig_b)


fig_b_b = plt.figure("GRAPH B_b")
ax = fig_b_b.add_subplot(111, projection='3d')
for i in range(neural_state.shape[1]):
    ax.plot(xs = range(neural_state.shape[0]), zs = neural_state[:,i], ys=np.repeat(i+1,neural_state.shape[0]), alpha=0.3)
    ax.plot(xs = range(neural_input.shape[0]), zs = neural_input[:,i], ys=np.repeat(i+1,neural_state.shape[0]))
plt.savefig("./graphs/GRAPH B_b (Neural Activity) [WiP]")
plt.close(fig_b_b)


# TODO: Maybe Wireframe:
fig_b_c = plt.figure("GRAPH B")
ax = fig_b_c.add_subplot(111, projection='3d')
for i in range(neural_state.shape[1]):
    ax.plot_wireframe(X = range(neural_state.shape[0]), Z = neural_state[:,i], Y=i+1)
# plt.plot(neural_input, alpha=0.3)
plt.savefig("./graphs/GRAPH B_C (Neural Activity) WIRE [WiP]")
plt.close(fig_b_c)


# TODO: Contour plots
# fig_b_d = plt.figure("GRAPH B_b")
# ax = fig_b_d.add_subplot(111, projection='3d')
# for i in range(neural_state.shape[1]):
#     ax.counter(X = range(neural_state.shape[0]), Z = neural_state[:,i], Y=i+1, alpha=0.3)
#     ax.counter(X = range(neural_input.shape[0]), Z = neural_input[:,i], Y=i+1)
# plt.savefig("./graphs/GRAPH B_b (Neural Activity) [WiP]")
# plt.close(fig_b_d)


## GRAPH C:
# keypress[2], sounds[3]
fig_c = plt.figure("GRAPH C")
for i in range(len(sr[2])):
    if sr[2][i,0] == 1: # keypress left
        plt.plot(i, sr[2][i,0]+1, 'gs', markersize=8)

    if sr[2][i, 1] == 1:  # keypress right
        plt.plot(i, sr[2][i, 0]-1, 'gs', markersize=8)

for i in range(len(sr[3])):
    if sr[3][i,0] == -1: # sound left
        plt.plot(i, sr[3][i,0]+1, 'yo', markersize=12, alpha=0.3)

    if sr[3][i, 1] == 1:  # sound right
        plt.plot(i, sr[3][i, 0]-1, 'yo', markersize=12, alpha=0.3)

plt.savefig("./graphs/GRAPH C (Keypress and Sound) [WiP]")
plt.close(fig_c)




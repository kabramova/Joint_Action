from SA_Evolution import *
from JA_Evolution import *

import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Info: http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

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


## Split in different Trials:
sl = sa_performance[0] # speed: slow, initial target-direction: left
sr = sa_performance[1] # speed: slow, initial target-direction: right
fl = sa_performance[2] # speed: fast, initial target-direction: left
fr = sa_performance[3] # speed: fast, initial target-direction: right

## GRAPH A:
tracker = sr[1][:,0] # trajectories[1], tracker: tracs[:,0]
target  = sr[1][:,1] # trajectories[1], target:  tracs[:,1]

plt.figure("GRAPH A")
plt.plot(tracker,'r', markersize=12, alpha=0.5)
plt.plot(target, 'g')
plt.savefig("./graphs/GRAPH A (POSITIONS) [WiP]")
plt.close()

## GRAPH B:
# neural_state[4]
sr[4].shape  # knoblin.Y
neural_state = sr[4]

# neural_input_L[5]
sr[5].shape  # knoblin.I
neural_input = sr[5]

#TODO: Turn plots, probably with zdir="x,y,u" and shift
# Info: http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

fig = plt.figure("GRAPH B")
ax = fig.add_subplot(111, projection='3d')
for i in range(neural_state.shape[1]):
    plt.plot(xs = range(neural_state.shape[0]), ys = neural_state[:,i], zs=i+1)
    plt.plot(xs = range(neural_input.shape[0]), ys = neural_input[:,i], zs=i+1, alpha=0.0)
# plt.plot(neural_input, alpha=0.3)
plt.savefig("./graphs/GRAPH B (Neural Activity) [WiP]")
plt.close(fig)

fig_b = plt.figure("GRAPH B_b")
ax = fig_b.add_subplot(111, projection='3d')
for i in range(neural_state.shape[1]):
    plt.plot(xs = range(neural_state.shape[0]), ys = neural_state[:,i], zs=i+1, alpha=0.3)
    plt.plot(xs = range(neural_input.shape[0]), ys = neural_input[:,i], zs=i+1)
plt.savefig("./graphs/GRAPH B_b (Neural Activity) [WiP]")
plt.close(fig_b)


#TODO: Maybe Wireframe:
fig = plt.figure("GRAPH B")
ax = fig.add_subplot(111, projection='3d')
for i in range(neural_state.shape[1]):
    Axes3D.plot_wireframe(X = range(neural_state.shape[0]), Y = neural_state[:,i], Z=i+1)
# plt.plot(neural_input, alpha=0.3)
plt.savefig("./graphs/GRAPH B (Neural Activity) [WiP]")
plt.close(fig)


## GRAPH C:
# keypress[2], sounds[3]
sr[2]
sr[3]

from Simulator import *

'''
a1 = Simulate()
a1.agent.show_innards()
a1.agent.show_innards(rounds=True)
a1.agent.Tau = a1.agent.Tau/a1.agent.Tau
a1.agent.W = a1.agent.W/a1.agent.W
a1.agent.WM = a1.agent.WM/a1.agent.WM # * np.array([2,1])
a1.agent.WV = a1.agent.WV/a1.agent.WV # * np.array([2,1])
a1.agent.Theta = a1.agent.Theta/a1.agent.Theta
a1.agent.show_innards(rounds=True)
a1.agent.position = np.array([10,50])
a1.agent.position_target = np.array([90,50])
a1.agent.Angle = 0
'''

a1 = Simulate(5000)
a1.run_and_plot()

## Agent 3 with specific weights


def a_run(sim = 500, h = 0.01, Angle = 0, pos_target = [50,11]):

    a3 = Simulate(sim)

    # Tau = 0.1
    a3.agent.Tau = (a3.agent.Tau/a3.agent.Tau)/10

    # Weights = 0
    a3.agent.W = a3.agent.W - a3.agent.W

    # Weights between 6&5 and between 2&3 = 1
    a3.agent.W[4,5] = 1
    a3.agent.W[2,1] = 1

    # switching neuron 1 on:
    a3.agent.W[4,0] = 1
    a3.agent.W[2,0] = -1

    # all nodes are recurrent
    # np.fill_diagonal(a3.agent.W,1)
    a3.agent.W[0,0] = 0   # ..., but node 1

    # Outer weights = 1, inner weights = 0
    a3.agent.WM = a3.agent.WM / a3.agent.WM *2
    a3.agent.WM[0] = a3.agent.WM[0] - a3.agent.WM[0]        # inner weights

    # Vision: Outer = 0.1, inner  = 10
    a3.agent.WV = ((a3.agent.WV / a3.agent.WV)) / 10        # outer weights to Neuron 2,6
    a3.agent.WV[1] = a3.agent.WV[1]/a3.agent.WV[1] *(10)    # inner weights to Neuron 1

    # Bias = 0
    a3.agent.Theta = a3.agent.Theta - a3.agent.Theta
    a3.agent.Theta[0] = .5

    # Show ingredients
    a3.agent.show_innards()

    # Position Agent and target
    a3.agent.position = np.array([50,10])
    a3.agent.position_target = pos_target

    # Set angle
    a3.agent.Angle = Angle

    # set timesteps
    a3.agent.h = h

    # Plot
    a3.run_and_plot()
    plt.plot(a3.agent.position_target[0], a3.agent.position_target[1], 'bs')

    return a3


# # These two run approx. the same path
# a_run(sim = 500, h = 0.01)
# a_run(sim =5000, h = 0.002832)
#
# ratio_sim = 5000/500
# ratio_h   = 0.002832/0.01
#
# a_run(sim =50000, h = 0.0008020224)


circle = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2,  7*np.pi/4]

for i in circle:
    a3 = a_run(sim = 5000, h = 0.01, Angle=i)



# TODO: Test and enhance the relevance of the visual input !!!

# - include "self._angle_to_target"
# - plus for both eyes.

# Agent_position = [50,10]
target = [ [30,11], [40,11], [45,11], [49,11],
           [50,11],
           [51,11], [55,11], [60,11], [70,11] ]

for i in target:
    a1 = a_run(sim = 1000, h = 0.01, Angle=np.pi/2, pos_target=i)


a_run(sim = 5000, h = 0.01, Angle=np.pi/2, pos_target=target[0])



n1 = a_run(1, Angle=np.pi/2, pos_target=[70,11]) # target right
n2 = a_run(1, Angle=np.pi/2, pos_target=[30,11]) # target left
plt.close()

n1.agent.Y
n2.agent.Y
n1.agent.visual_input()  # if target right then angle of left eye is bigger than of right eye
n2.agent.visual_input()  # if target right then angle of left eye is smaller than of right eye

n1.agent.show_innards()




n3 = a_run(1, Angle=np.pi/2, pos_target=[90,11]) # target very right
n4 = a_run(1, Angle=np.pi/2, pos_target=[51,11]) # target slightly right
n5 = a_run(1, Angle=np.pi/2, pos_target=[10,11]) # target very left
n6 = a_run(1, Angle=np.pi/2, pos_target=[49,11]) # target slightly left


n3.agent.visual_input()[0] - n3.agent.visual_input()[1]
n4.agent.visual_input()[0] - n4.agent.visual_input()[1]

n5.agent.visual_input()[0] - n5.agent.visual_input()[1]
n6.agent.visual_input()[0] - n6.agent.visual_input()[1]

n3.agent.I
n4.agent.I
n5.agent.I
n6.agent.I




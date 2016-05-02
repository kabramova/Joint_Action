from CTRNN import *
from Formulas import *

# Knoblich Jordan, 2003
# Target initial direction is random, but balanced (left, right)
# If target reaches the border, it abruptly turns back
# Three target turns during each trial
# Experiment: 3 blocks of 40 trials each.
# Impact of keypress was either low (velocity change 0.7° per second squared) or high (1.0° per second squared).
# The order of trials was randomized within each block.
#
# Implementation:

class Tracker:
    '''
    Initial tracker velocity = 0
    Tracker and target start at same position = 0
    '''

    def __init__(self):
        # trial, env_range will be globally announced by class Jordan
        self.position = 0
        self.velocity = 0

        self.distance_to_target = 0  # +/-, first, we try to let the network calculate this by its own.

        # TODO: Test whether the NN needs this information
        self.distance_to_bordeL = np.abs(env_range[0] - self.position)  # env_range will be globally announced by class Jordan
        self.distance_to_bordeR = np.abs(env_range[1] - self.position)

        self.trial = trial # either "fast" or "slow"


    def accelerate(self, input):
        '''
        Accelerates the tacker to either the left or the right
        Impact of keypress is either:
        - low velocity change 0.7° per second squared ["slow"]
        or
        - high (1.0° per second squared) ["fast"].
        :param input: either +1 or -1    (right or left)
        :return: update self.velocity
        '''
        if self.trial == "fast":
            acceleration = input*1
        else: # trial == "slow"
            acceleration = input*0.7

        # oldVelo = copy.copy(self.velocity)
        self.velocity += acceleration


    def movement(self):
        '''
        :return: update self.position
        '''
        self.position += self.velocity * h  # h will be globally announced in Agent (Knoblin)


    def dist_target(self, pos_target): # see comment self.distance_to_target (above)
        self.distance_to_target = pos_target - self.position


class Target:
    '''
    Target moves with constant velocity
    Tracker and target start at same position = 0
    Target velocity varied across the trials within each block.
    Target velocity was either slow (3.3° per second) or fast (4.3° per second).
    '''
    def __init__(self):
        # trial, env_range will be globally announced by class Jordan
        self.position = 0
        self.velocity = 4.3 if trial=="fast" else 3.3

        # TODO: Test whether the NN needs this information
        self.distance_to_bordeL = np.abs(env_range[0] - self.position)
        self.distance_to_bordeR = np.abs(env_range[1] - self.position)


    def movement(self):
        self.position += self.velocity * h  # h will be globally announced in Agent (Knoblin)


class Knoblin(CTRNN):
    '''
    Agent(s) alla Knoblich and Jordan (2003)

    Net-Input:
        - Position target, position tracker (minimal solution)
    Alternatively, three forms of inputs: (scaled solution):
        - distance tracker-target
        - distance tracker-boundaries
        - 2 different auditive stimuli

    2 different tones for the keys (left, right), duration = 100ms
    '''
    # TODO: Auditory Input (condition)

    def __init__(self):
        self.N_sensor = 2
        self.N_motor = 2

        # TODO: We could also apply symmetrical weights
        self.WM = randrange(self.W_RANGE, self.N_motor * 2, 1)
        self.WV = randrange(self.W_RANGE, self.N_sensor * 2, 1)

        # TODO: What is a reasonable number of neurons?
        super(self.__class__, self).__init__(number_of_neurons=6, timestep=0.01)

        global h
        h = self.h

    def press_right(self):
        return 1

    def press_left(self):
        return -1


    def visual_input(self, position_tracker, position_target):  # optional: distance_to_boarder
        '''
        Currently we just take the position of the tracker and the position of the target as input (minimal design).
        It's debatable, whether we want to tell the network directly, what the distance between Target and Agent is

        :param position_tracker, position_target: both informations come from the class Tracker & class Target
        :return:
        '''
        self.I[self.N-1] = self.WV[2] * position_tracker  # to left Neuron 6
        self.I[0] = np.sum(((self.WV[0] * position_tracker), (self.WV[1] * position_target))) # suppose to subtract the two inputs
        self.I[1] = self.WV[3] * position_target          # to right Neuron 2


    def motor_output(self):
        '''
        self.WM[0] : outer_weights_ML, N5_ML = [0]
        self.WM[1] : outer_weights_MR, N3_MR = [1]
        self.WM[2] : inner_weights_ML, N3_ML = [2]
        self.WM[3] : inner_weights_MR, N5_MR = [3]

        N3, round(N/2): = attach Motor-output down right (clockwise enumeration of network 1=Top)
        N5, N - round(N/2) + 2 : = attach Motor-output down left
        :return: output
        '''

        N3 = self.Y[round(self.N / 2) - 1]               # Neuron 3 (if N=6)
        N5 = self.Y[self.N - round(self.N / 2) + 2 - 1]  # Neuron 5 (if N=6)

        activation_left  = np.sum([N3 * self.WM[2], N5 * self.WM[0] ])
        activation_right = np.sum([N3 * self.WM[1], N5 * self.WM[3] ])

        # TODO: How to take constant input/output into account : e.g. constant output over threshold leads to key-press every 0.3 timesteps?
        # Threshold for output
        threshold = ...

        if activation_left > threshold:
            return self.press_left()

        if activation_right > threshold:
            return self.press_right()



class Jordan:
    # Task Environment
    def __init__(self, trial_speed, auditory_condition):

        self.trial = trial_speed
        global trial
        trial = self.trial

        self.condition = auditory_condition
        global condition
        condition = self.condition

        # TODO: Environment range either [-50,50] or [0,100]
        self.env_range = [-50, 50]
        global env_range
        env_range = self.env_range

from CTRNN import *
from Formulas import *

# Knoblich Jordan, 2003
# Target initial direction is random, but balanced (left, right)
# If target reaches the border, it abruptly turns back
# Three target turns during each trial
# Experiment: 3 blocks of 40 trials each.
# The order of trials was randomized within each block.
#
# Implementation:

class Tracker:
    '''
    Initial tracker velocity = 0
    Tracker and target start at same position = 0
    '''

    def __init__(self):
        self.position = 0
        self.velocity = 0
        # Timer for the emitted sound-feedback
        self.timer_sound_l = 0
        self.timer_sound_r = 0


    def accelerate(self, input):
        '''
        Accelerates the tacker to either the left or the right
        Impact of keypress is either:
        - low velocity change 0.7° per second squared ["slow"]
        or
        - high (1.0° per second squared) ["fast"].
        :param input: either -1 or +1    (left or right)
        :return: update self.velocity
        '''

        if input not in [-1,1]: raise ValueError("Input must be ∈ [-1;1]")

        if trial == "fast":         # trial will be globally announced by class Jordan
            acceleration = input*1
        else: # trial == "slow"
            acceleration = input*0.7

        # oldVelo = copy.copy(self.velocity)
        self.velocity += acceleration

        if condition==True: # condition will be globally announced by class Jordan
            self.set_timer(left_or_right=input)


    def set_timer(self, left_or_right):
        ''' Tone of 100-ms duration '''
        if left_or_right == -1:  # left
            self.timer_sound_l = 0.1
        elif left_or_right == 1: # right
            self.timer_sound_r = 0.1


    def movement(self):
        ''' Update self.position and self.timer(sound) '''
        self.position += self.velocity * h  # h will be globally announced in Agent (Knoblin)

        sound_output = [None,None]
        if condition == True:

            if self.timer_sound_l > 0:
                self.timer_sound_l -= h
                sound_output[0] = -1   # for auditory feedback

            if self.timer_sound_r > 0:
                self.timer_sound_r -= h
                sound_output[1] = 1  # for auditory feedback

        # TODO: how to deal with sound_output at next stage:
        return sound_output


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
        #TODO: randomise the initialization of direction (maybe in the JA_Simulator.py):
        self.velocity = 4.3 if trial=="fast" else 3.3

        # Distances to Boarders:
        self.distance_to_bordeL = np.abs(env_range[0] - self.position)
        self.distance_to_bordeR = np.abs(env_range[1] - self.position)


    def movement(self):
        self.direction()
        self.position += self.velocity * h  # h will be globally announced in Agent (Knoblin)

    def direction(self):
        #TODO: fine grained turning point:
        if self.distance_to_bordeR == 0 or self.distance_to_bordeL == 0:
            self.velocity *= -1


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
        self.N_visual_sensor = 2
        self.N_auditory_sensor = 2
        self.N_motor = 2

        super(self.__class__, self).__init__(number_of_neurons=8, timestep=0.01)

        # TODO: We could also apply symmetrical weights
        self.WM = randrange(self.W_RANGE, self.N_motor * 2, 1)          # Weights to Keypress (left,right)
        self.WV = randrange(self.W_RANGE, self.N_visual_sensor * 2, 1)  # Weights of visual input
        self.WA = randrange(self.W_RANGE, self.N_visual_sensor * 2, 1)  # Weights of auditory input, Keypress (left,right)

        global h
        h = self.h

        self.timer_motor_l = 0
        self.timer_motor_r = 0


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
        self.I[self.N-1] = self.WV[2] * position_tracker  # to left Neuron 8
        self.I[0] = np.sum(((self.WV[0] * position_tracker), (self.WV[1] * position_target))) # suppose to subtract the two inputs
        self.I[1] = self.WV[3] * position_target          # to right Neuron 2


    def auditory_input(self, input):  # optional: distance_to_boarder
        '''
        Currently we just take the position of the tracker and the position of the target as input (minimal design).
        It's debatable, whether we want to tell the network directly, what the distance between Target and Agent is

        :param input: Tone of either left(0), right(1) or both(2) keypress (0,1,2 coding, respectively)
        :return:
        '''

        if condition==True:  # condition will be globally announced by class Jordan

            if input not in [0, 1, 2]: raise ValueError("Input must be ∈ [0;1;2]")

            left_klick, right_klick = 1, 1

            if input == 0:
                self.I[self.N-2] = self.WA[0] * left_klick   # to left Neuron 7, left ear
                self.I[round(self.N / 2)] = self.WA[1] * left_klick

            elif input == 1:
                self.I[round(self.N / 2)] = self.WA[3] * right_klick
                self.I[2] = self.WA[2] * right_klick         # to right Neuron 3, right ear

            else: # input == 2:
                self.I[self.N - 2] = self.WA[0] * left_klick # to left Neuron 7, left ear
                self.I[round(self.N / 2)] = np.sum(((self.WA[1] * left_klick), (self.WA[3] * right_klick)))
                self.I[2] = self.WA[2] * right_klick         # to right Neuron 3, right ear


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

        N4 = self.Y[round(self.N / 2) - 1]               # Neuron 4 (if N=8)
        N6 = self.Y[self.N - round(self.N / 2) + 2 - 1]  # Neuron 6 (if N=8)

        activation_left  = np.sum([N4 * self.WM[2], N6 * self.WM[0] ])
        activation_right = np.sum([N4 * self.WM[1], N6 * self.WM[3] ])

        # TODO: How to take constant input/output into account : e.g. constant output over threshold leads to key-press every 0.5 timesteps?
        # Threshold for output
        threshold = ...

        if self.timer_motor_l > 0:
            self.timer_motor_l -= self.h
        if self.timer_motor_r > 0:
            self.timer_motor_r -= self.h

        if activation_left > threshold:
            if self.timer_motor_l == 0:
                self.timer_motor_l = 0.5
                return self.press_left()

        if activation_right > threshold:
            if self.timer_motor_r == 0:
                self.timer_motor_r = 0.5
                return self.press_right()



class Jordan:
    # Task Environment
    def __init__(self, trial_speed, auditory_condition):

        if trial_speed not in ["fast", "slow"]: raise ValueError("Must be either 'fast' or 'slow'")
        self.trial = trial_speed
        global trial
        trial = self.trial

        if auditory_condition not in [True, False]: raise ValueError("Must be either True or False")
        self.condition = auditory_condition
        global condition
        condition = self.condition

        # TODO: Environment range either [-50,50] or [0,100]
        self.env_range = [-50, 50]
        global env_range
        env_range = self.env_range

    def auditive_feedback(self):
        # Tone of 100-ms duration
        pass

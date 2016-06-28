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
    """
    Initial tracker velocity = 0
    Tracker and target start at same position = 0
    """

    def __init__(self):
        self.position = 0
        self.velocity = 0
        # Timer for the emitted sound-feedback
        self.timer_sound_l = 0
        self.timer_sound_r = 0

    def accelerate(self, inputs):
        """
        Accelerates the tacker to either the left or the right
        Impact of keypress is either:
        - low velocity change 0.7° per second squared ["slow"]
        or
        - high (1.0° per second squared) ["fast"].
        :param inputs: either -1 or +1    (left or right)
        :return: update self.velocity
        """

        if inputs[0] not in [-1, 0] or inputs[1] not in [0, 1]: raise ValueError("inputs must be ∈ [-1; 1]")

        inputs = np.array(inputs)

        if trial == "fast":     # trial will be globally announced by class Jordan
            v = 1.0
        elif trial == "slow":
            v = 0.7

        acceleration = np.dot(np.array([v, v]).transpose(), inputs)

        self.velocity += acceleration

        if condition:              # condition will be globally announced by class Jordan
            self.set_timer(left_or_right=inputs)

    def set_timer(self, left_or_right):
        """ Tone of 100-ms duration """
        if left_or_right[0] == -1:  # left
            self.timer_sound_l = 0.1
        if left_or_right[1] == 1:   # right
            self.timer_sound_r = 0.1

    def movement(self):
        """ Update self.position and self.timer(sound) """

        self.position += self.velocity * h  # h will be globally announced in Agent (Knoblin)

        # Tacker does not continue moving, when at the edges of the environment.
        if self.position < env_range[0]:
            self.position = env_range[0]
        if self.position > env_range[1]:
            self.position = env_range[1]

        sound_output = [0, 0]

        if self.timer_sound_l > 0:
            self.timer_sound_l -= h
            sound_output[0] = 1   # for auditory feedback

        if self.timer_sound_r > 0:
            self.timer_sound_r -= h
            sound_output[1] = 1  # for auditory feedback

        return sound_output


class Target:
    """
    Target moves with constant velocity
    Tracker and target start at same position = 0
    Target velocity varied across the trials within each block.
    Target velocity was either slow (3.3° per second) or fast (4.3° per second).
    """

    def __init__(self):
        # trial, env_range will be globally announced by class Jordan
        self.position = 0
        self.velocity = 4.3 if trial == "fast" else 3.3

    def distance_to_border(self):
        """ Returns the distances of the Target to each boarder of the environment """
        distance_to_border_l = np.abs(env_range[0] - self.position)
        distance_to_border_r = np.abs(env_range[1] - self.position)
        return [distance_to_border_l, distance_to_border_r]

    def direction(self):
        """ Target turns direction if its crucially close to one of the boarders """
        distance_left_right = self.distance_to_border()
        if any(distance < np.abs(self.velocity * h) for distance in distance_left_right):
            self.velocity *= -1

    def movement(self):
        self.direction()
        self.position += self.velocity * h  # h will be globally announced in Agent (Knoblin)


class Knoblin(CTRNN):
    """
    Agent(s) alla Knoblich and Jordan (2003)

    Net-Input:
        - Position target, position tracker (minimal solution)
    Alternatively, three forms of inputs: (scaled solution):
        - distance tracker-target
        - distance tracker-boundaries
        - 2 different auditive stimuli

    2 different tones for the keys (left, right), duration = 100ms
    """

    def __init__(self):
        self.N_visual_sensor = 2
        self.N_auditory_sensor = 2
        self.N_motor = 2

        super(self.__class__, self).__init__(number_of_neurons=8, timestep=0.01)

        # TODO: We could also apply symmetrical weights
        # Weights to Keypress (left,right):
        self.WM = randrange(self.W_RANGE, self.N_motor * 2, 1)
        # Weights of visual input:
        self.WV = randrange(self.W_RANGE, self.N_visual_sensor * 2, 1)
        # Weights of auditory input, Keypress (left,right):
        self.WA = randrange(self.W_RANGE, self.N_auditory_sensor * 2, 1)

        global h
        h = self.h

        self.timer_motor_l = 0
        self.timer_motor_r = 0

    def press_left(self):
            return -1

    def press_right(self):
            return 1

    def visual_input(self, position_tracker, position_target):  # optional: distance_to_border
        """
        Currently we just take the position of the tracker and the position of the target as input (minimal design).
        It's debatable, whether we want to tell the network directly, what the distance between Target and Agent is.

        :param position_tracker, position_target: both information come from the class Tracker & class Target
        """
        # TODO: run with different input
        self.I[self.N-1] = self.WV[2] * position_tracker  # to left Neuron 8
        # suppose to subtract the two inputs:
        self.I[0] = np.sum(((self.WV[0] * position_tracker), (self.WV[1] * position_target)))
        self.I[1] = self.WV[3] * position_target          # to right Neuron 2

        # print("Visual Input: \n", self.I[[self.N-1, 0, 1]])

    def auditory_input(self, sound_input):
        """
        :param sound_input: Tone(s) induced by left click and/or right click: coming from class Tracker.movement()
        """

        if sound_input[0] not in [1, 0] or sound_input[1] not in [0, 1]:
            raise ValueError("Input must be ∈ [0; 1]")

        left_click, right_click = sound_input[0], sound_input[1]

        # to left Neuron 7, left ear   (if N=8):
        self.I[self.N-2] = self.WA[0] * left_click
        # to right Neuron 3, right ear (if N=8):
        self.I[2] = self.WA[2] * right_click
        # to middle Neuron 5:
        self.I[round(self.N / 2)] = np.sum(((self.WA[1] * left_click), (self.WA[3] * right_click)))

        # if any(i > 0 for i in sound_input):
        #     print("Auditory Input: \n", self.I[[self.N-2, 2, round(self.N / 2)]])

    def motor_output(self):
        """
        "Optimal performance was computed under the assumption of a maximal keypress rate of 5 keypresses per second" (Knoblich & Jordan, 2002))
        The assumption by Knoblich & Jordan does not necessarily cover with real performance of participants. More than 2-3 clicks per second are
        highly unlikely for controlled acceleration.
        We set timer to 0.5. That means we have max. 2 clicks per time-unit, but simultaneously clicking with other hand is possible.

        self.WM[0] : outer_weights_ML, N5_ML = [0]
        self.WM[1] : outer_weights_MR, N3_MR = [1]
        self.WM[2] : inner_weights_ML, N3_ML = [2]
        self.WM[3] : inner_weights_MR, N5_MR = [3]

        N3, round(N/2): = attach Motor-output down right (clockwise enumeration of network 1=Top)
        N5, N - round(N/2) + 2 : = attach Motor-output down left
        :return: output
        """

        n4 = self.Y[round(self.N / 2) - 1]               # Neuron 4 (if N=8)
        n6 = self.Y[self.N - round(self.N / 2) + 2 - 1]  # Neuron 6 (if N=8)

        activation_left = np.sum([n4 * self.WM[2], n6 * self.WM[0]])
        activation_right = np.sum([n4 * self.WM[1], n6 * self.WM[3]])

        # Update timer:
        if self.timer_motor_l > 0:
            self.timer_motor_l -= self.h
        if self.timer_motor_r > 0:
            self.timer_motor_r -= self.h

        # Threshold is arbitrary. Evolution should make use of any kind.
        threshold = 0           # Threshold for output
        activation = [0, 0]     # Activation is zero

        # We set timer to 0.5. That means we have max. 2 clicks per time-unit
        if activation_left > threshold:
            if self.timer_motor_l <= 0:
                self.timer_motor_l = 0.5
                activation[0] = self.press_left()   # press() will only return something if timer == 0.

        if activation_right > threshold:
            if self.timer_motor_r <= 0:
                self.timer_motor_r = 0.5
                activation[1] = self.press_right()  # press() will only return something if timer == 0.

        return activation

    def show_innards(self, rounds=False):
        if not rounds:
            print("Tau:\n", self.Tau,
                  "\n \n weights:\n", self.W,
                  "\n \n Motor weights:\n", self.WM,
                  "\n \n Vision weights:\n", self.WV,
                  "\n \n Audio weights:\n", self.WA,
                  "\n \n Biases:\n", self.Theta)
        else:
            print("Tau:\n", np.round(self.Tau, 2),
                  "\n \n weights:\n", np.round(self.W, 2),
                  "\n \n Motor weights:\n", np.round(self.WM, 2),
                  "\n \n Vision weights:\n", np.round(self.WV, 2),
                  "\n \n Audio weights:\n", np.round(self.WA, 2),
                  "\n \n Biases:\n", np.round(self.Theta, 2))


class Jordan:
    # Task Environment
    def __init__(self, trial_speed, auditory_condition):

        if trial_speed not in ["fast", "slow"]:
            raise ValueError("Must be either 'fast' or 'slow'")
        self.trial = trial_speed

        if auditory_condition not in [True, False]:
            raise ValueError("Must be either True or False")
        self.condition = auditory_condition

        # [-20,20] is a plausible Screen size(40cm) with a visual angle of approx.28 degrees (check with angle_velo2())
        self.env_range = [-20, 20]  # TODO: Environment range either [-20,20] or [0,40]

        self.globalization()

    def globalization(self):
        global trial
        trial = self.trial

        global condition
        condition = self.condition

        global env_range
        env_range = self.env_range

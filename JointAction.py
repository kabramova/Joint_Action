# Knoblich Jordan, 2003
#
# Participants in front of monitor, distance = 80 cm
# 2 different tones for the keys (left, right), duration = 100ms
# Tracker and target start at same position
# Initial tracker velocity = 0
# Target moves with constant velocity
# Target initial direction is random, but balanced (left, right)
# If target reaches the border, it abruptly turns back
# Three target turns during each trial
#
# Experiment: 3 blocks of 40 trials each.
# Target velocity and impact of keypress varied across the trials within each block.
# Target velocity was either slow (3.3° per second) or fast (4.3° per second).
# Impact of keypress was either low (velocity change 0.7° per second squared) or high (1.0° per second squared).
# The order of trials was randomized within each block.
#
# Implementation:
#
# - three forms of inputs
#     - distance tracker-target
#     - distance tracker-boundaries
#     - 2 different auditive stimuli
# alternatively: just position target, position tracker
#


class Tracker:

    def __init__(self, position):
        self.position = position
        self.velocity = 0

    def accelerate(self, input):
        '''

        :param input: should be either +1 or -1 (right or left)
        :return:
        '''
        pass

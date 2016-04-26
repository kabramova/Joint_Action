

# Knoblich Jordan, 2003
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

    def accelerate(self):
        pass

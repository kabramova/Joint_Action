from JointAction import *


class JA_Simulation:
# Joint Action Simulation:

    def __init__(self, simlength=1000):
        self.simlength = simlength

    def setup(self, trial_speed="slow", auditory_condition=False):
        self.environment = Jordan(trial_speed=trial_speed, auditory_condition=auditory_condition)
        self.knoblin = Knoblin()
        self.tracker = Tracker()
        self.target = Target()


    def run(self):
        for i in range(self.simlength):
            self.target.movement()
            self.tracker.movement()
            self.knoblin.visual_input(position_tracker=self.target.position, position_target=self.target.position)
            self.knoblin.auditory_input(input= ...)
            self.tracker.accelerate(input = self.knoblin.motor_output())







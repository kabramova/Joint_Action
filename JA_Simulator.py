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

            sound_output = self.tracker.movement()

            self.knoblin.visual_input(position_tracker=self.tracker.position, position_target=self.target.position)

            if condition == True: # condition will be globally announced by class Jordan (self.environment)
                self.knoblin.auditory_input(input= sound_output)

            if self.knoblin.timer_motor_l <= 0 or self.knoblin.timer_motor_r <= 0: # this is a bit redundant (see e.g.) Knoblin.press_left(), but more computational efficient
                self.tracker.accelerate(input = self.knoblin.motor_output())

            # TODO: Fitness, Trajectories, Evolution(Genome)







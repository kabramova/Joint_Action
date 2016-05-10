from JointAction import *


class JA_Simulation:
# Joint Action Simulation:

    def __init__(self, simlength=2789):
        # Withe simlength=2789, Target turns 3times during each trial (with regard to Knoblich & Jordan, 2003)
        self.simlength = simlength

    def setup(self, trial_speed="slow", auditory_condition=False):
        self.environment = Jordan(trial_speed=trial_speed, auditory_condition=auditory_condition)
        self.knoblin = Knoblin()
        self.globalization()
        self.tracker = Tracker()
        self.target = Target()


    def globalization(self):   # for a certain reason I have to add this here a second time.
        global trial
        trial = self.environment.trial

        global condition
        condition = self.environment.condition

        global env_range
        env_range = self.environment.env_range

        global h
        h = self.knoblin.h


    def run(self):

        fitness_curve = []

        for i in range(self.simlength):

            # print("Timestep:",i+1)

            # 1) Target movement
            self.target.movement()
            # plt.plot(i, self.target.position, "ro")

            # 2) Tracker movement
            sound_output = self.tracker.movement()
            # plt.plot(i, self.tracker.position, "bo")

            # 3) Agent sees:
            self.knoblin.visual_input(position_tracker=self.tracker.position, position_target=self.target.position)

            # 4) Agent hears:
            if condition == True: # condition will be globally announced by class Jordan (self.environment)
                self.knoblin.auditory_input(sound_input = sound_output)

            # 5) Update agent's neural system
            self.knoblin.next_state()

            # print("State of Neurons:\n", self.knoblin.Y)

            # 5) Agent reacts:
            if self.knoblin.timer_motor_l <= 0 or self.knoblin.timer_motor_r <= 0: # this is a bit redundant (see e.g.) Knoblin.press_left(), but more computational efficient
                activation = self.knoblin.motor_output()
                # if any(act > 0 for act in np.abs(activation)): print("Activation:", activation)
                self.tracker.accelerate(input = activation)

            # 6) Fitness tacking:
            fitness_curve.append(self.fitness())

        # 7) Overall fitness:
        print("Average distance to Target (Fitness:)", np.mean(fitness_curve))
        return np.mean(fitness_curve)


    def fitness(self):
        return np.abs(self.target.position - self.tracker.position)






# j1 = JA_Simulation(simlength=2789)
# j1.setup(trial_speed="fast", auditory_condition=True)
# j1.run()
# print("Target Position:", j1.target.position)
# print("Tracker Position:", j1.tracker.position)
# print("Tracker Velocity:", j1.tracker.velocity)
# print("Tracker Timer_L:", j1.tracker.timer_sound_l)
# print("Tracker Timer_R:",j1.tracker.timer_sound_r)

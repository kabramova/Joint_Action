from JointAction import *


class JA_Simulation:
# Joint Action Simulation:

    def __init__(self, simlength=2789):
        # Withe simlength=2789, Target turns 3times during each trial (with regard to Knoblich & Jordan, 2003)
        self.knoblin = Knoblin()
        self.simlength = simlength

    def setup(self, trial_speed="slow", auditory_condition=False):
        self.environment = Jordan(trial_speed=trial_speed, auditory_condition=auditory_condition)
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
        output = np.mean(fitness_curve)

        return output


    def run_and_plot(self):
        # TODO: how to deal with globally announced variables (see globalization())

        fitness_curve = []

        positions = np.zeros((self.simlength,2))
        keypress = np.zeros((self.simlength,2))
        activation = None # necessary for tracking of keypress
        if condition ==True:
            sounds = np.zeros((self.simlength,2))

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

            positions[i,:] = [self.tracker.position, self.target.position] # save positions

            # 4) Agent hears:
            if condition == True: # condition will be globally announced by class Jordan (self.environment)
                self.knoblin.auditory_input(sound_input = sound_output)
                sounds[i,:] = sound_output # save sound_output

            # 5) Update agent's neural system
            self.knoblin.next_state()

            # print("State of Neurons:\n", self.knoblin.Y)

            # 5) Agent reacts:
            if self.knoblin.timer_motor_l <= 0 or self.knoblin.timer_motor_r <= 0: # this is a bit redundant (see e.g.) Knoblin.press_left(), but more computational efficient
                activation = self.knoblin.motor_output()
                # if any(act > 0 for act in np.abs(activation)): print("Activation:", activation)
                self.tracker.accelerate(input = activation)
            else:
                keypress[i,:] = activation if activation != None else [0,0]

            # 6) Fitness tacking:
            fitness_curve.append(self.fitness())

        # 7) Overall fitness:
        print("Average distance to Target (Fitness:)", np.mean(fitness_curve))
        # TODO: Question is, whether this output is usefull
        output = [np.mean(fitness_curve)]

        output.append(positions)
        output.append(keypress)
        if condition == True:
            output.append(sounds)
        print("Output contains trajectories, keypress and sounds(if applicable)")


        ## PLOT and save current state of the system:
        # TODO: include condition & trialspeed
        # Create Folder for images:
        time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs("./Animation/{}.Animation".format(time))

        for i in range(self.simlength):

            ticker = 10 # just plot every 10th (x-th) state.
            if i % ticker == 0:
            # With a simlength of 2789 the resulting gif-animation is approx. 11sec long (25frames/sec)
            # we can change the animation length by changing the modulo here [i%x].

            plt.figure(figsize=(10, 6), dpi=80)

            plt.plot(positions[i, 0], 0, 'ro', markersize=12, alpha=0.5)    # Tracker
            plt.plot(positions[i, 1], 0, 'go')                              # Target

            if keypress[i, 0] == -1:
                plt.plot(-10, -4, 'bs', markersize=16)                      # keypress left
            if keypress[i, 1] == 1:
                plt.plot( 10, -4, 'bs', markersize=16)                      # keypress right

            if condition==True:
                if sounds[i,0] == 1:
                    plt.plot(-10, -3, 'yo', markersize=24, alpha=0.3)       # sound left
                if sounds[i, 1] == 1:
                    plt.plot( 10, -3, 'yo', markersize=24, alpha=0.3)       # sound right

            # Define boarders
            plt.xlim(-25, 25)
            plt.ylim(-5, 5)

            # Print Fitnesss in Plot
            plt.annotate(xy=[0, 4], xytext=[0, 4], s="fitness = {}".format(output[0]))

            plt.savefig('./Animation/{}.Animation/animation{}.png'.format(time, int(i/ticker) ) )

            plt.close()


        return output


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

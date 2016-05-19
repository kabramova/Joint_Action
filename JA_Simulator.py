from JointAction import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



class JA_Simulation:
# Joint Action Simulation:

    def __init__(self, simlength=2789):
        # Withe simlength=2789, Target turns 3times during each trial (with regard to Knoblich & Jordan, 2003)
        self.knoblin = Knoblin()
        self.simlength = simlength
        self.runs = 0               # to count how many runs the agent made

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


    def reset_neural_system(self):
        '''Sets all activation to zero'''
        self.knoblin.Y             = np.matrix(np.zeros((self.knoblin.N, 1)))
        self.knoblin.I             = np.matrix(np.zeros((self.knoblin.N, 1)))
        self.knoblin.timer_motor_l = 0
        self.knoblin.timer_motor_r = 0
        # Alternatively for class JA_Evolution:
        # self.knoblin = Knoblin()
        # self.implement_genome(genome_string=self.genome)


    def run(self):

        self.reset_neural_system()  # All inner neural states = 0

        fitness_curve = []

        self.runs += 1

        for i in range(self.simlength):

            # if i%200 ==0: print("Timestep:",i+1)  # Print every now and then a timestep

            # 1) Target movement
            self.target.movement()

            # 2) Tracker movement
            sound_output = self.tracker.movement()

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
        print("{} trial, Sound {}: Average distance to Target (Fitness:) {}".format(trial, condition, np.round(np.mean(fitness_curve),3)))
        output = np.round(np.mean(fitness_curve),3)

        return output


    def run_and_plot(self):
        # TODO: how to deal with globally announced variables (see globalization())

        self.reset_neural_system() # All inner neural states = 0

        fitness_curve = []

        positions = np.zeros((self.simlength,2))
        keypress = np.zeros((self.simlength,2))
        if condition ==True:
            sounds = np.zeros((self.simlength,2))

        print("Sound condition:\t {} \n"
              "Trial speed:\t {}".format(condition,trial))

        for i in range(self.simlength):

            # print("Timestep:",i+1)

            # 1) Target movement
            self.target.movement()

            # 2) Tracker movement
            sound_output = self.tracker.movement()

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

            # 6) Agent reacts:
            if self.knoblin.timer_motor_l <= 0 or self.knoblin.timer_motor_r <= 0:  # this is a bit redundant (see e.g.) Knoblin.press_left(), but more computational efficient
                activation = self.knoblin.motor_output()
                if any(act > 0 for act in np.abs(activation)): print("Activation:", activation)
                self.tracker.accelerate(input = activation)
                keypress[i, :] = activation

            # 7) Fitness tacking:
            fitness_curve.append(self.fitness())

        # 8) Overall fitness:
        print("Average distance to Target (Fitness:)", np.round(np.mean(fitness_curve),3))
        output = [np.round(np.mean(fitness_curve),3)]

        output.append(positions)
        output.append(keypress)
        if condition == True:
            output.append(sounds)
        print("Output contains fitness[0], trajectories[1], keypress[2] and sounds[3](if applicable)")


        ## PLOT and save current state of the system:
        # TODO: include condition & trialspeed
        # Create Folder for images:
        time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs("./Animation/{}.Animation".format(time))

        ticker = 10  # just plot every 10th (x-th) state.
        counter_img = 0
        counter_sec = 0

        for i in np.arange(0,self.simlength+1,ticker):

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
                    plt.plot(-10, -3.9, 'yo', markersize=24, alpha=0.3)       # sound left
                if sounds[i, 1] == 1:
                    plt.plot( 10, -3.9, 'yo', markersize=24, alpha=0.3)       # sound right

            # Define boarders
            plt.xlim(-20, 20)
            plt.ylim(-5, 5)

            # Print Fitnesss, time and conditions in Plot
            plt.annotate(xy=[0, 4], xytext=[0, 4], s="fitness = {}".format(output[0])) # Fitness

            # Updated time-counter:
            if counter_img==25:
                counter_sec += 1
            counter_img = counter_img + 1 if counter_img < 25 else 1

            plt.annotate(xy=[-15, 4], xytext=[-15, 4], s="Time = {}:{}sec".format(str(counter_sec).zfill(2), str(counter_img).zfill(2))) # Time
            print("Time = {}:{}sec".format(str(counter_sec).zfill(2), str(counter_img).zfill(2))) # Time)

            plt.annotate(xy=[-15, 3.5], xytext=[-15, 3.5], s="{} Trial".format(trial))                 # trial
            plt.annotate(xy=[-15, 3.0], xytext=[-15, 3.0], s="Sound Condition: {}".format(condition))  # condition

            plt.savefig('./Animation/{}.Animation/animation{}.png'.format(time, str(int(i/ticker)).zfill(len(str(int(self.simlength/ticker))))))


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

from JointAction import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

"""
__author__  = Simon Hofmann"
__credits__ = ["Simon Hofmann", "Katja Abramova", "Willem Zuidema"]
__version__ = "1.0.1"
__date__ "2016"
__maintainer__ = "Simon Hofmann"
__email__ = "simon.hofmann@protonmail.com"
__status__ = "Development"
"""


class JA_Simulation:
    """ Joint Action Simulation (Joint Task): """

    def __init__(self, auditory_condition, symmetrical_weights=False, simlength=2789):
        self.knoblin_l = Knoblin(symmetrical_weights=symmetrical_weights)
        self.knoblin_r = Knoblin(symmetrical_weights=symmetrical_weights)
        # With simlength=2789, Target turns 3times during each fast trial (with regard to Knoblich & Jordan, 2003)
        # With simlength=3635, Target turns 3times during each slow trial, will be changed in setup
        self.simlength = simlength
        self.condition = auditory_condition
        self.runs = 0               # to count how many runs the agent made

        # Will be initialized within setup()
        self.environment = []
        self.tracker = []
        self.target = []

    def setup(self, trial_speed, simlength_scalar=1):
        """
        Setup the experiment: Tracker, target, environment.
        Prepares for different trial speeds.
        """
        self.environment = Jordan(trial_speed=trial_speed, auditory_condition=self.condition)

        if self.environment.trial == "slow" and self.simlength != 3635:
            self.simlength = int(np.round(3635*simlength_scalar))  # Target needs more time to make 3 turns

        elif self.environment.trial == "fast" and self.simlength != 2789:
            self.simlength = int(np.round(2789*simlength_scalar))  # Target needs less time to make 3 turns

        self.globalization()

        self.tracker = Tracker()
        self.target = Target()

    def globalization(self):  # for a certain reason I have to add this here a second time.
        global trial
        trial = self.environment.trial

        global env_range
        env_range = self.environment.env_range

        global h
        h = self.knoblin_l.h  # == self.knoblin_r.h

    def reset_neural_system(self):
        """Sets all activation to zero"""
        self.knoblin_l.Y = np.matrix(np.zeros((self.knoblin_l.N, 1)))
        self.knoblin_l.I = np.matrix(np.zeros((self.knoblin_l.N, 1)))
        self.knoblin_l.timer_motor_l = 0
        self.knoblin_l.timer_motor_r = 0

        self.knoblin_r.Y = np.matrix(np.zeros((self.knoblin_r.N, 1)))
        self.knoblin_r.I = np.matrix(np.zeros((self.knoblin_r.N, 1)))
        self.knoblin_r.timer_motor_l = 0
        self.knoblin_r.timer_motor_r = 0
        # Alternatively for class JA_Evolution:
        # self.knoblin = Knoblin()
        # self.implement_genome(genome_string=self.genome)

    def run(self):

        self.reset_neural_system()  # All inner neural states = 0

        fitness_curve = []

        self.runs += 1

        for i in range(self.simlength):

            # if i % 200 == 0: print("Timestep:", i+1)  # Print every now and then a timestep

            # 1) Target movement
            self.target.movement()

            # 2) Tracker movement
            sound_output = self.tracker.movement()

            # 3) Agent sees:
            self.knoblin_l.visual_input(position_tracker=self.tracker.position, position_target=self.target.position)
            self.knoblin_r.visual_input(position_tracker=self.tracker.position, position_target=self.target.position)

            # 4) Agent hears:
            if self.condition:  # condition will be globally announced by class Jordan (self.environment)
                self.knoblin_l.auditory_input(sound_input=sound_output)
                self.knoblin_r.auditory_input(sound_input=sound_output)

            # 5) Update agent's neural systems
            self.knoblin_l.next_state()
            self.knoblin_r.next_state()

            # 5) Agents react:
            if self.knoblin_l.timer_motor_l <= 0 or self.knoblin_l.timer_motor_r <= 0 or \
                            self.knoblin_r.timer_motor_l <= 0 or self.knoblin_r.timer_motor_r <= 0:
                # this is a bit redundant (see e.g.) Knoblin.press_left(), but more computational efficient
                activation_l = self.knoblin_l.motor_output()
                activation_r = self.knoblin_r.motor_output()
                activation = [activation_l[0], activation_r[1]]
                # if any(act > 0 for act in np.abs(activation)): print("Activation:", activation)
                self.tracker.accelerate(inputs=activation)

            # 6) Fitness tacking:
            fitness_curve.append(self.fitness())

        # 7) Overall fitness:
        # print("{} trial, Sound {}: Average distance to Target (Fitness): {}".format(trial, self.condition,
        # np.round(np.mean(fitness_curve),3)))
        output = np.round(np.mean(fitness_curve), 3)

        return output

    def run_and_plot(self, lesion=False):

        self.reset_neural_system()  # All inner neural states = 0

        fitness_curve = []

        positions = np.zeros((self.simlength, 2))
        keypress = np.zeros((self.simlength, 2))
        sounds = np.zeros((self.simlength, 2))
        neural_state_l = np.zeros((self.simlength, self.knoblin_l.N))
        neural_state_r = np.zeros((self.simlength, self.knoblin_r.N))
        neural_input_l = np.zeros((self.simlength, self.knoblin_l.N))
        neural_input_r = np.zeros((self.simlength, self.knoblin_r.N))

        print("Sound condition:\t {} \n"
              "Trial speed:\t {}".format(self.condition, trial))

        for i in range(self.simlength):

            # print("Timestep:",i+1)

            # 1) Target movement
            self.target.movement()

            # 2) Tracker movement
            sound_output = self.tracker.movement()

            # 3) Agents see:
            # if not lesion or lesion and i < self.simlength/2:  # (Lesion cuts input in second half of simulation)
            if not lesion:  # cut all input
                self.knoblin_l.visual_input(position_tracker=self.tracker.position, position_target=self.target.position)
                self.knoblin_r.visual_input(position_tracker=self.tracker.position, position_target=self.target.position)
            else:  # No input anymore
                self.knoblin_l.visual_input(position_tracker=0, position_target=0)
                self.knoblin_r.visual_input(position_tracker=0, position_target=0)

            positions[i, :] = [self.tracker.position, self.target.position]  # save positions

            # 4) Agents hear:
            if self.condition:  # condition will be globally announced by class Jordan (self.environment)
                # if not lesion or lesion and i < self.simlength / 2:  # (Lesion cuts input in second half of simulation)
                if not lesion:  # cut all input
                    self.knoblin_l.auditory_input(sound_input=sound_output)
                    self.knoblin_r.auditory_input(sound_input=sound_output)
                else:  # No input anymore
                    self.knoblin_l.auditory_input(sound_input=[0, 0])
                    self.knoblin_r.auditory_input(sound_input=[0, 0])

                sounds[i, :] = sound_output  # save sound_output

            # 5) Update agents' neural systems
            self.knoblin_l.next_state()
            self.knoblin_r.next_state()
            neural_state_l[i, :] = self.knoblin_l.Y.transpose()
            neural_state_r[i, :] = self.knoblin_r.Y.transpose()
            neural_input_l[i, :] = self.knoblin_l.I.transpose()
            neural_input_r[i, :] = self.knoblin_r.I.transpose()

            # 6) Agents react:
            if self.knoblin_l.timer_motor_l <= 0 or self.knoblin_l.timer_motor_r <= 0 or self.knoblin_r.timer_motor_l <= 0 \
                    or self.knoblin_r.timer_motor_r <= 0:
                # this is a bit redundant (see e.g.) Knoblin.press_left(), but more computational efficient
                activation_l = self.knoblin_l.motor_output()
                activation_r = self.knoblin_r.motor_output()
                activation = [activation_l[0], activation_r[1]]

                # if any(act > 0 for act in np.abs(activation)): print("Activation:", activation)
                self.tracker.accelerate(inputs=activation)
                keypress[i, :] = activation

            # 7) Fitness tacking:
            fitness_curve.append(self.fitness())

        # 8) Overall fitness:
        print("{} trial, Sound {}: Average distance to Target (Fitness): {}"
              .format(trial, self.condition, np.round(np.mean(fitness_curve), 3)))

        output = [np.round(np.mean(fitness_curve), 3)]

        output.append(positions)
        output.append(keypress)
        output.append(sounds)
        output.append(neural_state_l)
        output.append(neural_state_r)
        output.append(neural_input_l)
        output.append(neural_input_r)
        # print("Output contains fitness[0], trajectories[1], keypress[2], sounds[3],
        # neural_state_l[4], neural_state_r[5], neural_input_r[6], neural_input_r[7]")

        # PLOT and save current state of the system:
        # Create Folder for images:
        times = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs("./Animation/{}.Animation.Sound_{}.{}_trial".format(times, self.condition, trial))

        ticker = 10         # just plot every 10th (x-th) state.
        counter_img = 0
        counter_sec = 0

        # Region Borders
        upper_bound = self.environment.env_range[1]
        lower_bound = self.environment.env_range[0]
        screen_width = upper_bound - lower_bound
        region_width = screen_width / 3
        right_border = 0 + region_width / 2
        left_border = 0 - region_width / 2

        # Set initial Target Direction:
        direction = "left" if positions[1, 1] < 0 else "right"  # Test Target Position after first update

        for i in np.arange(0, self.simlength, ticker):

            # For Fast Trials: with a simlength of 2789 the resulting gif-animation is approx. 11sec long (25frames/sec)
            # & for Slow Trials: with a simlength of 3635.
            # we can change the animation length by changing the modulo here [i%x].

            plt.figure(figsize=(10, 6), dpi=80)

            plt.plot(np.repeat(left_border, len(range(-5, 5))), range(-5, 5), "--", c="grey", alpha=0.2)   # Region Left
            plt.plot(np.repeat(right_border, len(range(-5, 5))), range(-5, 5), "--", c="grey", alpha=0.2)  # Region Right

            plt.plot(positions[i, 0], 0, 'ro', markersize=12, alpha=0.5)     # Tracker
            plt.plot(positions[i, 1], 0, 'go')                               # Target

            if any(keypress[i:i+ticker, 0] == -1):
                plt.plot(-10, -4, 'bs', markersize=16)                       # keypress left
            if any(keypress[i:i+ticker, 1] == 1):
                plt.plot(10, -4, 'bs', markersize=16)                        # keypress right

            if self.condition:
                if any(sounds[i:i+ticker, 0] == 1):
                    plt.plot(-10, -3.9, 'yo', markersize=24, alpha=0.3)      # sound left
                if any(sounds[i:i+ticker, 1] == 1):
                    plt.plot(10, -3.9, 'yo', markersize=24, alpha=0.3)       # sound right

            # Define boarders
            plt.xlim(-20, 20)
            plt.ylim(-5, 5)

            # Remove y-Axis
            plt.yticks([])

            # Print Fitnesss, time and conditions in Plot
            plt.annotate(xy=[0, 4], xytext=[0, 4], s="Trial Fitness: {}".format(output[0]))  # Fitness

            # Updated time-counter:
            if counter_img == 25:
                counter_sec += 1
                print("{}% ready".format(np.round((i / self.simlength) * 100, 2)))  # gives feedback how much is plotted already.

            counter_img = counter_img + 1 if counter_img < 25 else 1

            # Update simulation time:
            sim_msec = i if i < 100 else i % 100
            sim_sec = int(i*h)  # or int(i/100)

            plt.annotate(xy=[-15, 4], xytext=[-15, 4], s="{}:{}sec (Real Time)".format(str(counter_sec).zfill(2),
                                                                                       str(counter_img).zfill(2)))         # Real Time

            plt.annotate(xy=[-15, 3.5], xytext=[-15, 3.5], s="{}:{}sec (Simulation Time)".format(str(sim_sec).zfill(2),
                                                                                                 str(sim_msec).zfill(2)))  # Simulation Time

            plt.annotate(xy=[0, 3.5], xytext=[0, 3.5], s="Initial Target Direction: {}".format(direction))                 # Target Direction
            plt.annotate(xy=[0, 3.0], xytext=[0, 3.0], s="{} Trial".format(trial))                                         # trial
            plt.annotate(xy=[-15, 3.0], xytext=[-15, 3.0], s="Sound Condition: {}".format(self.condition))                 # condition

            plt.savefig('./Animation/{}.Animation.Sound_{}.{}_trial/'
                        'animation{}.png'.format(times,
                                                 self.condition,
                                                 trial,
                                                 str(int(i/ticker)).zfill(len(str(int(self.simlength/ticker))))))

            plt.close()

        return output

    def fitness(self):
        assert self.target and self.target, "Envorinoment must be setup() and run() first."
        return np.abs(self.target.position - self.tracker.position)

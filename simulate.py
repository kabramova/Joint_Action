import numpy as np
# from profilestats import profile


class Simulation:
    def __init__(self, width, step_size, velocities, impacts, startperiod, condition=False):
        self.width = width  # [-20, 20]
        self.step_size = step_size  # typically h, how fast things are happening in the simulation
        self.trials = self.create_trials(velocities, impacts)
        distance = (width[1]-width[0]) * 3
        self.sim_length = [int(distance/abs(trial[0])/self.step_size) for trial in self.trials]  # simulation length depends on target velocity
        self.condition = condition  # is it a sound condition?
        self.startperiod = startperiod  # the period of time at the beginning of the trial in which the target stays still

    @staticmethod
    def create_trials(velocities, impacts):
        """
        Create a list of trials the environment will run.
        :return: 
        """
        trials = [(x, y) for x in velocities for y in impacts]
        return trials

    # @profile(print_stats=10)
    def run_trials(self, agent, trials, savedata=False):
        """
        An evaluation function that accepts an agent and returns a real number representing
        the performance of that parameter vector on the task. Here the task is the Knoblich and Jordan task.

        :param agent: an agent with a CTRNN brain and particular anatomy
        :param trials: a list of trials to perform
        :param savedata: should the trial data be saved
        :return: fitness
        """

        trial_data = dict()
        trial_data['fitness'] = []
        trial_data['target_pos'] = [None] * len(trials)
        trial_data['tracker_pos'] = [None] * len(trials)

        if savedata:
            trial_data['tracker_v'] = [None] * len(trials)
            trial_data['brain_state'] = [None] * len(trials)
            trial_data['input_output'] = [None] * len(trials)
            trial_data['keypress'] = [None] * len(trials)

        for i in range(len(trials)):
            target = Target(trials[i][0], self.step_size)
            tracker = Tracker(trials[i][1], self.step_size)
            # set initial state in specified range
            state_range = [-0.5, 0.5]
            # state_range = [0, 0]
            agent.brain.randomize_state(state_range)

            trial_data['target_pos'][i] = np.zeros((self.sim_length[i] + self.startperiod, 1))
            trial_data['tracker_pos'][i] = np.zeros((self.sim_length[i] + self.startperiod, 1))

            if savedata:
                trial_data['tracker_v'][i] = np.zeros((self.sim_length[i] + self.startperiod, 1))
                trial_data['brain_state'][i] = np.zeros((self.sim_length[i] + self.startperiod, agent.brain.N))
                trial_data['input_output'][i] = np.zeros((self.sim_length[i] + self.startperiod, 12))
                trial_data['keypress'][i] = np.zeros((self.sim_length[i] + self.startperiod, 2))

            if self.startperiod > 0:
                # don't move the target
                for j in range(self.startperiod):
                    agent.visual_input(tracker.position, target.position)
                    agent.brain.euler_step()
                    activation = agent.motor_output()
                    trial_data['target_pos'][i][j] = target.position
                    trial_data['tracker_pos'][i][j] = tracker.position
                    if savedata:
                        trial_data['tracker_v'][i][j] = tracker.velocity
                        trial_data['brain_state'][i][j] = agent.brain.Y
                        trial_data['keypress'][i][j] = activation
                        trial_data['input_output'][i][j] = np.hstack((agent.VW, agent.AW, agent.MW))

            for j in range(self.startperiod, self.sim_length[i] + self.startperiod):

                # 1) Target movement
                target.movement(self.width)

                # 2) Agent sees
                agent.visual_input(tracker.position, target.position)

                # 3) Agents clicks button
                sound_output = tracker.movement(self.width)

                # 4) Agent hears
                if self.condition:
                    agent.auditory_input(sound_output)

                trial_data['target_pos'][i][j] = target.position
                trial_data['tracker_pos'][i][j] = tracker.position
                if savedata:
                    trial_data['tracker_v'][i][j] = tracker.velocity
                    trial_data['brain_state'][i][j] = agent.brain.Y

                # 5) Update agent's neural system
                agent.brain.euler_step()

                # 6) Agent reacts
                activation = agent.motor_output()
                tracker.accelerate(activation)

                if savedata:
                    trial_data['keypress'][i][j] = activation
                    trial_data['input_output'][i][j] = np.hstack((agent.VW, agent.AW, agent.MW))

            # 6) Fitness tacking:
            # trial_data['fitness'].append(1 - (np.sum(np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i])) /
            #                                   (2*self.width[1]*self.sim_length[i])))

            cap_distance = 10
            scores = np.clip(-1/cap_distance * np.abs(trial_data['target_pos'][i] - trial_data['tracker_pos'][i]) + 1, 0, 1)
            trial_data['fitness'].append(np.mean(scores))

        return trial_data


class Target:
    """
    Target moves with constant velocity and starts from the middle.
    Target velocity varies across the trials within each block: either slow (3.3° per second) or fast (4.3° per second).
    """

    def __init__(self, velocity, step_size):
        self.position = 0
        self.velocity = velocity
        self.step_size = step_size

    def reverse_direction(self, border_range, future_pos):
        """
        Reverse target direction if going beyond the border.
        :param border_range: border positions of the environment
        :param future_pos: predicted position at the next time step
        :return: 
        """
        if ((self.velocity > 0) and (future_pos > border_range[1])) or \
                ((self.velocity < 0) and (future_pos < border_range[0])):
            self.velocity *= -1

    def movement(self, border_range):
        future_pos = self.position + self.velocity * self.step_size
        self.reverse_direction(border_range, future_pos)
        self.position += self.velocity * self.step_size


class Tracker:
    """
    Tracker moves as a result of its set velocity and can accelerate based on agent button clicks.
    It starts in the middle of the screen and with initial 0 velocity.
    """

    def __init__(self, impact, step_size, condition=False):
        self.position = 0
        self.velocity = 0
        self.impact = impact  # how much acceleration is added by button click
        self.step_size = step_size
        # Timer for the emitted sound-feedback
        self.condition = condition  # is it a sound condition?
        self.timer_sound_l = 0
        self.timer_sound_r = 0

    def movement(self, border_range):
        """ Update self.position and self.timer(sound) """

        self.position += self.velocity * self.step_size  # h will be globally announced in Agent (Knoblin)

        # Tacker does not continue moving, when at the edges of the environment.
        if self.position < border_range[0]:
            self.position = border_range[0]
        if self.position > border_range[1]:
            self.position = border_range[1]

        sound_output = [0, 0]

        if self.timer_sound_l > 0:
            self.timer_sound_l -= self.step_size
            sound_output[0] = 1   # for auditory feedback

        if self.timer_sound_r > 0:
            self.timer_sound_r -= self.step_size
            sound_output[1] = 1  # for auditory feedback

        return sound_output

    def accelerate(self, inputs):
        """
        Accelerates the tracker to either the left or the right
        Impact of keypress is either:
        - low velocity change 0.7° per second squared ["slow"]
        or
        - high (1.0° per second squared) ["fast"].
        :param inputs: an array of size two with values of -1 or +1 (left or right)
        :return: update self.velocity
        """
        acceleration = np.dot(np.array([self.impact, self.impact]).T, inputs)
        self.velocity += acceleration

        if self.condition:
            self.set_timer(inputs)

    def set_timer(self, left_or_right):
        """ Tone of 100-ms duration """
        if left_or_right[0] == -1:  # left
            self.timer_sound_l = 0.1
        if left_or_right[1] == 1:   # right
            self.timer_sound_r = 0.1


class Agent:
    """
    This is a class that implements agents in the simulation. Agents' brains are CTRNN, but they also have
    a particular anatomy and a connection to external input and output.
    """
    def __init__(self, network):
        self.brain = network
        self.n_visual_sensors = 2
        self.n_auditory_sensors = 2
        self.n_motors = 2
        self.VW = np.random.uniform(self.brain.w_range[0], self.brain.w_range[1], (self.n_visual_sensors * 2))
        self.AW = np.random.uniform(self.brain.w_range[0], self.brain.w_range[1], (self.n_auditory_sensors * 2))
        self.MW = np.random.uniform(self.brain.w_range[0], self.brain.w_range[1], (self.n_motors * 2))
        self.gene_range = [0, 1]
        self.genotype = self.make_genotype_from_params()
        self.fitness = 0

        self.timer_motor_l = 0
        self.timer_motor_r = 0
        self.crossover_points = [11, 22, 33, 44, 55, 66, 77, 88, 92, 96]

        # n_modules = 4 * self.brain.N + 3  # 4 params per neuron and input-output modules
        # crossover_points = [i*(3 + self.brain.N) for i in range(1, self.brain.N+1)]
        # crossover_points.extend([crossover_points[-1]+i*len(self.VW) for i in range(4)])

    def __eq__(self, other):
        if np.all(self.genotype == other.genotype):
            return True

    def make_genotype_from_params(self):
        """
        Combine all parameters and reshape into a single vector
        :return: [Tau_n1, G_n1, Theta_n1, W_n1..., all visual w, all auditory w, all motor w] 
        """
        # TODO should genotype modules be defined in terms of nodes whose parameters are kept together in crossover?
        # return [self.Tau, self.G, self.Theta, self.W]
        tau = self.linmap(self.brain.Tau, self.brain.tau_range, self.gene_range)
        # skip G in evolution
        # g = self.linmap(self.brain.G, self.brain.g_range, [0, 1])
        theta = self.linmap(self.brain.Theta, self.brain.theta_range, self.gene_range)
        w = self.linmap(self.brain.W.T, self.brain.w_range, self.gene_range)
        vw = self.linmap(self.VW, self.brain.w_range, self.gene_range)
        aw = self.linmap(self.AW, self.brain.w_range, self.gene_range)
        mw = self.linmap(self.MW, self.brain.w_range, self.gene_range)

        stacked = np.vstack((tau, theta, w))
        flattened = stacked.reshape(stacked.size, order='F')
        genotype = np.hstack((flattened, vw, aw, mw))
        return genotype

    def make_params_from_genotype(self, genotype):
        genorest, self.VW, self.AW, self.MW = np.hsplit(genotype, [88, 92, 96])
        unflattened = genorest.reshape(3+self.brain.N, self.brain.N, order='F')
        self.brain.Tau, self.brain.G, self.brain.Theta, self.brain.W = (np.squeeze(a) for a in np.vsplit(unflattened, [1, 2, 3]))
        self.brain.W = self.brain.W.T

        genorest, vw, aw, mw = np.hsplit(genotype, [80, 84, 88])
        self.VW = self.linmap(vw, self.gene_range, self.brain.w_range)
        self.AW = self.linmap(aw, self.gene_range, self.brain.w_range)
        self.MW = self.linmap(mw, self.gene_range, self.brain.w_range)

        unflattened = genorest.reshape(2+self.brain.N, self.brain.N, order='F')
        tau, theta, w = (np.squeeze(a) for a in np.vsplit(unflattened, [1, 2]))
        self.brain.Tau = self.linmap(tau, self.gene_range, self.brain.tau_range)
        self.brain.Theta = self.linmap(theta, self.gene_range, self.brain.theta_range)
        self.brain.W = self.linmap(w, self.gene_range, self.brain.w_range).transpose()

    def visual_input(self, position_tracker, position_target):
        """
        The visual input to the agent
        :param position_tracker: absolute position of the tracker 
        :param position_target: absolute position of the target
        :return:
        """
        self.brain.I[7] = self.VW[0] * position_tracker  # to n8
        self.brain.I[1] = self.VW[1] * position_target  # to n2
        # self.brain.I[0] = np.sum([self.VW[2] * position_target, self.VW[3] * position_tracker])  # to n1
        self.brain.I[0] = self.VW[2] * position_target + self.VW[3] * position_tracker  # to n1

    def auditory_input(self, sound_input):
        """
        The auditory input to the agent
        :param sound_input: Tone(s) induced by left and/or right click
        """
        left_click, right_click = sound_input[0], sound_input[1]

        self.brain.I[6] = self.AW[0] * left_click  # to n7
        self.brain.I[2] = self.AW[1] * right_click  # to n3
        # self.brain.I[4] = np.sum([self.AW[2] * left_click, self.AW[3] * right_click])  # to n5
        self.brain.I[4] = self.AW[2] * left_click + self.AW[3] * right_click  # to n5

    def motor_output(self):
        """
        The motor output of the agent
        :return: output
        """
        # Set activation threshold
        activation = [0, 0]  # Initial activation is zero
        threshold = 0  # Threshold for output

        n4 = self.brain.Y[3]  # from n4
        n6 = self.brain.Y[5]  # from n6

        # activation_left = np.sum([n4 * self.MW[0], n6 * self.MW[2]])
        # activation_right = np.sum([n4 * self.MW[1], n6 * self.MW[3]])

        activation_left = n4 * self.MW[0] + n6 * self.MW[2]
        activation_right = n4 * self.MW[1] + n6 * self.MW[3]

        # Update timer:
        if self.timer_motor_l > 0:
            self.timer_motor_l -= self.brain.step_size
        if self.timer_motor_r > 0:
            self.timer_motor_r -= self.brain.step_size

        # We set timer to 0.5. That means we have max. 2 clicks per time-unit
        if activation_left > threshold:
            if self.timer_motor_l <= 0:
                self.timer_motor_l = 0.5  # reset the timer
                activation[0] = -1   # set left activation to -1 to influence velocity to the left

        if activation_right > threshold:
            if self.timer_motor_r <= 0:
                self.timer_motor_r = 0.5
                activation[1] = 1  # set right to one to influence velocity to the right

        return activation

    @staticmethod
    def linmap(vin, rin, rout):
        """
        Map a vector between 2 ranges.
        :param vin: input vector to be mapped
        :param rin: range of vin to map from
        :param rout: range to map to
        :return: mapped output vector
        :rtype np.ndarray
        """
        a = rin[0]
        b = rin[1]
        c = rout[0]
        d = rout[1]
        return ((c + d) + (d - c) * ((2 * vin - (a + b)) / (b - a))) / 2

from JA_Simulator import *
import pickle

class JA_Evolution(JA_Simulation):

    def __init__(self, pop_size=111):

        super(self.__class__, self).__init__(simlength=2789) # self.knoblin, self.simlength

        self.genome = self.create_genome(Knoblin=self.knoblin)

        self.generation = 0

        self.pop_size = pop_size

        self.pop_list = self.__create_pop_list(pop_size)

        self.filename = ""


    def __create_pop_list(self, pop_size):
        '''
         :param pop_size: Amount of individuals per Population
         :return: list of agents
         '''

        poplist = np.zeros((pop_size, np.size(self.genome) + 2))

        for i in range(pop_size):
            poplist[i, 0] = i + 1                                   # enumerate the list
            # poplist[i, 1]                                         = fitness, is initially zero
            poplist[i, 2:] = self.genome.transpose()                # the current genome will be stored
            self.knoblin = Knoblin()                                # Create new agent
            self.genome = self.create_genome(Knoblin=self.knoblin)  # ... and its genome

        return poplist


    def create_genome(self, Knoblin):

        A = np.reshape(Knoblin.W,      (Knoblin.W.size,       1))
        G = np.reshape(Knoblin.WM,     (Knoblin.WM.size,      1))
        T = np.reshape(Knoblin.WV,     (Knoblin.WV.size,      1))
        X = np.reshape(Knoblin.WA,     (Knoblin.WA.size,      1))
        C = np.reshape(Knoblin.Theta,  (Knoblin.Theta.size,   1))
        U = np.reshape(Knoblin.Tau,    (Knoblin.Tau.size,     1))

        return np.concatenate((A, G, T, X, C, U))


    def implement_genome(self, genome_string):

        assert genome_string.size == self.genome.size, "Genome has not the right size"

        A = self.knoblin.W.size
        G = self.knoblin.WM.size
        T = self.knoblin.WV.size
        X = self.knoblin.WA.size
        C = self.knoblin.Theta.size
        U = self.knoblin.Tau.size

        W       = genome_string[:A]
        WM      = genome_string[A:A + G]
        WV      = genome_string[A + G:A + G + T]
        WA      = genome_string[A + G + T:A + G + T + X]
        Theta   = genome_string[A + G + T + X:A + G + T + X + C]
        Tau     = genome_string[A + G + T + X + C:A + G + T + X + C + U]

        self.knoblin.W = np.reshape(W, (self.knoblin.N, self.knoblin.N))
        self.knoblin.WM = WM
        self.knoblin.WV = WV
        self.knoblin.WA = WA
        self.knoblin.Theta = Theta
        self.knoblin.Tau = Tau

        # Update the self.genome:
        if not isinstance(genome_string, np.matrix):
            genome_string = np.matrix(genome_string).transpose()

        self.genome = genome_string


    def reset_neural_system(self):
        '''Sets all activation to zero'''
        self.knoblin.Y             = np.matrix(np.zeros((self.knoblin.N, 1)))
        self.knoblin.I             = np.matrix(np.zeros((self.knoblin.N, 1)))
        self.knoblin.timer_motor_l = 0
        self.knoblin.timer_motor_r = 0
        # Alternatively:
        # self.knoblin = Knoblin()
        # self.implement_genome(genome_string=self.genome)


    def run_trials(self):

        fitness_per_trials = []

        for trial_speed in ["slow", "fast"]:
            for auditory_condition in [False, True]:
                for init_target_direction in [-1, 1]:  # left(-1) or right(1)

                    self.setup(trial_speed=trial_speed, auditory_condition=auditory_condition)
                    self.target.velocity *= init_target_direction

                    # Run trial:
                    fitness = self.run(track=False)[0]
                    fitness_per_trials.append(fitness)

                    self.reset_neural_system()

        fitness = np.mean(fitness_per_trials)
        print("Average fitness over all trials:", fitness)

        return fitness


    def _run_population(self):

        for i, string in enumerate(self.pop_list):
            if string[1] == 0:  # run only if fitness is no evaluated yet
                genome = string[2:]
                self.knoblin = Knoblin()
                self.implement_genome(genome_string=genome)
                # Run all trials an save fitness in pop_list:
                self.pop_list[i, 1] = self.run_trials()

        self.pop_list = copy.copy(mat_sort(self.pop_list, index=1)) # sorts the pop_list, best agents on top


    def gen_code(self):
        gens = OrderedDict([("A", self.agent.W.size),
                            ("G", self.agent.WM.size),
                            ("T", self.agent.WV.size),
                            ("X", self.agent.WA.size),
                            ("C", self.agent.Theta.size),
                            ("U", self.agent.Tau.size)])
        return gens


    def _reproduction(self, mutation_var=.02):
        '''
        Combination of asexual (fitness proportionate selection (fps)) sexual reproduction
            Minimal population size = 10
            1) Takes the two best agents and copy them in new population.
            2) Based on pop_size, creates 2-10 children (parents: two best agents)
                - Since step 1) we have a chance of genetic crossover of 100%.
                - we use whole sections of the genome for the crossover (e.g. all W, or all Thetas)
                - 20% of population size and max. 10
            3) Fitness proportionate selection of 30% (+ 1/2 fill up)
            4) Fill with randomly created agents, 30% (+ 1/2 fill up)
            5) All but the first two best agents will fall under a mutation with a variance of .02 (default)
                - time constant: τ (tau) in range [1, 10]
                - weights: w (weights of interneurons, sensory and motor neurons) in range [-13, 13]
                - bias: θ (theta) in range [-13, 13]

        :param mutation_var: 0.02 by default, turned out to be better.
        :return: self.pop_list = repopulated list (new_population)
        '''

        gens = self.gen_code()

        new_population = np.zeros(self.pop_list.shape)


        # 1) Takes the two best agents and copy them in new population.
        n_parents = 2
        new_population[0:n_parents,:] = copy.copy(self.pop_list[(0,1),:])

        # 2) Based on pop_size, creates 2-10 children (parents: two best agents)
        n_children = self.pop_size*0.2 if self.pop_size*0.2 < 10 else 10

        for n in range(n_children):
            new_population[2+n,2:] = copy.copy(self.pop_list[0,2:])

            ## Crossover of a whole genome section of the second parent:
            choice = np.random.choice([gen for gen in gens])  # Random choice of a section in genome

            index = 0  # indexing the section in whole genome string
            for gen in gens:
                index += gens[gen]
                if gen == choice:
                    break
            index += 2  # leaves the number and fitness of agent out (new_population[:,(0,1)])

            new_population[2+n, (index - gens[choice]):index] = copy.copy(self.pop_list[1, (index - gens[choice]):index])  # crossover from second parent

            # TODO: Test: self.agent.PARAMETER (depending on choice)

        # 3) Fitness proportionate selection of 30% (+ 1/2 fill up)

        # Define the number of agents via fps & via random instantiation
        n_family = n_parents + n_children
        n_fps = self.pop_size*0.3
        n_random = self.pop_size*0.3

        if (self.pop_size - (n_family + n_fps + n_random))!= 0:
            rest = self.pop_size - (n_family + n_fps + n_random) # rest has to be filled up
            if rest%2>0:  # if rest is odd
                n_fps += (rest+1)/2
                n_random += (rest-1)/2
            else:         # if rest is even
                n_fps += rest/2
                n_random += rest/2

        # Algorithm for fitness proportionate selection:
        # (Source: http://stackoverflow.com/questions/298301/roulette-wheel-selection-algorithm/320788#320788)

        fitness = copy.copy(self.pop_list[:, 1])
        fitness = 1 - normalize(fitness)  # sign is correct, apparently

        total_fitness = sum(fitness)
        relative_fitness = [f / total_fitness for f in fitness]

        probs = [sum(relative_fitness[:i + 1]) for i in range(len(relative_fitness))]

        for n in range(n_family, n_family+n_fps):   # or range(n_family, self.pop_size-n_random-1)
            r = np.random.random()        # random sample of continuous uniform distribution [0,1)

            for (i, individual) in enumerate(self.pop_list):
                if r <= probs[i]:
                    new_population[n, :] = individual
                    break


        # 4) Fill with randomly created agents, 30% (+ 1/2 fill up)
        n_fitfamily = n_family + n_fps
        for n in range(n_fitfamily, n_fitfamily+n_random):
            self.knoblin = Knoblin()                                    # Create random new agent
            self.genome = self.create_genome(Knoblin = self.knoblin)    # ... and its genome
            new_population[n, 2:] = self.genome.transpose()


        # 5) All but the first two best agents will fall under a mutation with a variance of .02 (default)

        AGTXC = sum(gens.values()) - gens["U"]  # sum of all gen-sizes, except Tau
        U = gens["U"]  # == self.agent.Tau.size

        mu, sigma = 0, np.sqrt(mutation_var)  # mean and standard deviation

        for i in range(n_parents, n_fitfamily):  # we start with the 3rd agent and end with the agents via fps, rest is random, anyways.

            mutation_AGTXC = np.random.normal(mu, sigma, AGTXC)
            mutation_U = np.random.normal(mu, sigma, U)

            AGTXC_mutated = new_population[i, 2: AGTXC+2] + mutation_AGTXC

            AGTXC_mutated[AGTXC_mutated > self.agent.W_RANGE[1]] = self.agent.W_RANGE[1]  # Replace values beyond the range with max.range
            AGTXC_mutated[AGTXC_mutated < self.agent.W_RANGE[0]] = self.agent.W_RANGE[0]  # ... or min.range (T_RANGE = W.RANGE =[-13, 13])

            new_population[i, 2: AGTXC+2] = AGTXC_mutated

            U_mutated = new_population[i, (AGTXC + 2):] + mutation_U

            U_mutated[U_mutated > self.agent.TAU_RANGE[1]] = self.agent.TAU_RANGE[1]  # Replace values beyond the range with max.range
            U_mutated[U_mutated < self.agent.TAU_RANGE[0]] = self.agent.TAU_RANGE[0]  # ... or min.range (TAU_RANGE = [1, 10])

            new_population[i, (AGTXC + 2):] = U_mutated


        # Reset enumeration and fitness (except first two agents)
        new_population[:, 0] = range(1, self.pop_size+1)
        new_population[n_parents:, 1] = 0

        self.pop_list = new_population


    def run_evolution(self, generations, mutation_var=.02):

        save = save_request()

        # Run evolution:
        Fitness_progress = np.zeros((generations, 6))

        for i in range(generations):

            # Create new Generation
            self._reproduction(mutation_var)

            # Evaluate fitness of each member
            self._run_population()

            Fitness_progress[i, 1:] = np.round(self.pop_list[0:5, 1], 2)

            self.generation += 1

            Fitness_progress[i, 0] = self.generation

            print(Fitness_progress[i, 1:], "Generation", self.generation)

        # Save in external file:
        if save:

            self.filename = "sim{}.mut{}.Gen{}-{}.JA.single".format(self.simlength,
                                                                    mutation_var,
                                                                    self.generation - generations + 1,
                                                                    self.generation)

            pickle.dump(self.pop_list, open('Poplist.{}'.format(self.filename), 'wb'))
            pickle.dump(np.round(Fitness_progress, 2), open('Fitness_progress.{}'.format(self.filename), 'wb'))

            print('Evolution terminated. pop_list saved \n'
                  '(Filename: "Poplist.{}")'.format(self.filename))
        else:
            print('Evolution terminated. \n'
                  '(Caution: pop_list is not saved in external file)')


    def reimplement_population(self, filename=None, Plot=False):

        if filename is None:
            if self.filename != "":
                filename = self.filename
                print("Reimplements its own pop_list file")
            else:
                raise ValueError("No file to reimplement")

        # Reimplement: pop_list, simlength, Generation
        self.pop_list = pickle.load(open('Poplist.{}'.format(filename), 'rb'))

        self.simlength = int(filename[filename.find('m') + 1: filename.find('.')])  # depends on filename

        fitness_progress = pickle.load(open('Fitness_progress.{}'.format(filename), 'rb'))
        self.generation = int(fitness_progress[-1, 0])

        if Plot:
            # here we plot the fitness progress of all generation
            plt.figure()
            for i in range(1, fitness_progress.shape[1])
                plt.plot(fitness_progress[:, i])

            # Here we plot the trajectory of the best agent:
            self.plot_pop_list()
            print("Plot the best agent")

            global n  # this is needed for self.close()
            n = 2

    # TODO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def plot_pop_list(self, n_knoblins=1):

        global n
        n = n_knoblins

        for i in range(n_knoblins):
            self.reset_neural_system()
            self.implement_genome(genome_string=self.pop_list[i,2:])

            for j in range(self.simlength):
                # TODO: def run_and_plot() (Trajectories, Gif)
                # TODO: how to deal with globally announced variables (see globalization())

                output = self.run(track=True)
                output...


                plt.figure()

                plt.plot()
                plt.close()


        print(np.round(self.pop_list[0:n_knoblins, 0:3], 2))
        if n_knoblins > 1:
            print("Close all Windows with close()")


    def close(self):
        for j in range(n):  # n is from the global variable of plot_pop_list()/reimplement_population()
            plt.close()


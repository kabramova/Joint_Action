from SA_Simulator import *
import pickle

class SA_Evolution(SA_Simulation):

    def __init__(self, auditory_condition, pop_size=111):

        super(self.__class__, self).__init__(auditory_condition, simlength=2789) # self.knoblin, self.simlength, self.condition

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


    def run_trials(self):

        fitness_per_trials = []

        for trial_speed in ["slow", "fast"]:
            for init_target_direction in [-1, 1]:  # left(-1) or right(1)

                self.setup(trial_speed=trial_speed)
                self.target.velocity *= init_target_direction

                # Run trial:
                fitness = self.run()
                fitness_per_trials.append(fitness)


        fitness = np.mean(fitness_per_trials)
        # print("Average fitness over all 8 trials:", np.round(fitness,2))

        return fitness


    def _run_population(self):

        first_runs = False

        for i, string in enumerate(self.pop_list):
            if string[1] == 0:  # run only if fitness is no evaluated yet
                genome = string[2:]
                self.knoblin = Knoblin()
                self.implement_genome(genome_string=genome)

                # Run all trials an save fitness in pop_list:
                ticker = 10
                if i % ticker == 0 or i < 10:  # this way because it ignores the two first spots in pop_list, since they run already.
                    if i % ticker == 0:
                        fill = i + ticker if i <= self.pop_size - ticker else self.pop_size
                        first_runs = True
                        print("Generation {}: Run trials for Agents {}-{}".format(self.generation, i + 1, fill))
                    if i < 10 and first_runs == False:
                        fill = ticker
                        first_runs = True
                        print("Fitness of first agents were already evaluated")
                        print("Generation {}: Run trials for Agents {}-{}".format(self.generation, i + 1, fill))


                self.pop_list[i, 1] = self.run_trials()

        self.pop_list = copy.copy(mat_sort(self.pop_list, index=1)) # sorts the pop_list, best agents on top


    def gen_code(self):
        gens = OrderedDict([("A", self.knoblin.W.size),
                            ("G", self.knoblin.WM.size),
                            ("T", self.knoblin.WV.size),
                            ("X", self.knoblin.WA.size),
                            ("C", self.knoblin.Theta.size),
                            ("U", self.knoblin.Tau.size)])
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
        n_children = int(np.round(self.pop_size*0.2) if np.round(self.pop_size*0.2) < 10 else 10)

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


        # 3) Fitness proportionate selection of 30% (+ 1/2 fill up)

        # Define the number of agents via fps & via random instantiation
        n_family = n_parents + n_children
        n_fps    = int(np.round(self.pop_size*0.3))
        n_random = int(np.round(self.pop_size*0.3))

        if (self.pop_size - (n_family + n_fps + n_random))!= 0:
            rest = self.pop_size - (n_family + n_fps + n_random) # rest has to be filled up
            if rest%2>0:  # if rest is odd
                n_fps += int((rest+1)/2)
                n_random += int((rest-1)/2)
            else:         # if rest is even
                n_fps += int(rest/2)
                n_random += int(rest/2)

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
        U = gens["U"]  # == self.knoblin.Tau.size

        mu, sigma = 0, np.sqrt(mutation_var)  # mean and standard deviation

        for i in range(n_parents, n_fitfamily):  # we start with the 3rd agent and end with the agents via fps, rest is random, anyways.

            mutation_AGTXC = np.random.normal(mu, sigma, AGTXC)
            mutation_U = np.random.normal(mu, sigma, U)

            AGTXC_mutated = new_population[i, 2: AGTXC+2] + mutation_AGTXC

            AGTXC_mutated[AGTXC_mutated > self.knoblin.W_RANGE[1]] = self.knoblin.W_RANGE[1]  # Replace values beyond the range with max.range
            AGTXC_mutated[AGTXC_mutated < self.knoblin.W_RANGE[0]] = self.knoblin.W_RANGE[0]  # ... or min.range (T_RANGE = W.RANGE =[-13, 13])

            new_population[i, 2: AGTXC+2] = AGTXC_mutated

            U_mutated = new_population[i, (AGTXC + 2):] + mutation_U

            U_mutated[U_mutated > self.knoblin.TAU_RANGE[1]] = self.knoblin.TAU_RANGE[1]  # Replace values beyond the range with max.range
            U_mutated[U_mutated < self.knoblin.TAU_RANGE[0]] = self.knoblin.TAU_RANGE[0]  # ... or min.range (TAU_RANGE = [1, 10])

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
            if i != 0:
                self._reproduction(mutation_var)

            # Evaluate fitness of each member
            self._run_population()

            Fitness_progress[i, 1:] = np.round(self.pop_list[0:5, 1], 2) # saves fitness progress for the five best agents

            self.generation += 1

            Fitness_progress[i, 0] = self.generation

            print("Generation {}: Fitness (5 best Agents): {}".format(self.generation, Fitness_progress[i, 1:]))

        # Save in external file:
        if save:
            self.filename = "Gen{}-{}.popsize{}.mut{}.sound_cond={}.JA.single(Fitness{})".format(self.generation - generations + 1,
                                                                                                 self.generation,
                                                                                                 self.pop_size,
                                                                                                 mutation_var,
                                                                                                 self.condition,
                                                                                                 np.round(self.pop_list[0,1],2))

            pickle.dump(self.pop_list, open('Poplist.{}'.format(self.filename), 'wb'))
            pickle.dump(np.round(Fitness_progress, 2), open('Fitness_progress.{}'.format(self.filename), 'wb'))

            print('Evolution terminated. pop_list saved \n'
                  '(Filename: "Poplist.{}")'.format(self.filename))
        else:
            print('Evolution terminated. \n'
                  '(Caution: pop_list is not saved in external file)')


    def reimplement_population(self, filename=None, Plot=False):

        if filename is None:
            if self.filename == "":
                raise ValueError("No file to reimplement")
            else:
                print("Reimplements its own pop_list file")
        else:
            self.filename = filename

        # Reimplement: pop_list, condition, Generation
        self.pop_list = pickle.load(open('Poplist.{}'.format(self.filename), 'rb'))
        self.pop_size = self.pop_list.shape[0]

        assert self.filename.find("False") != -1 or self.filename.find("True") != -1, "Condition is unknown (please add to filename (if known)"
        self.condition = False if self.filename.find("False") != -1 and self.filename.find("True") == -1 else True

        fitness_progress = pickle.load(open('Fitness_progress.{}'.format(self.filename), 'rb'))
        self.generation = int(fitness_progress[-1, 0])

        # self.setup(trial_speed="fast") # Trial speed is arbitrary. This command is needed to globally announce variables

        if Plot:

            # here we plot the fitness progress of all generation
            plt.figure()
            for i in range(1, fitness_progress.shape[1]):
                plt.plot(fitness_progress[:, i])
                plt.ylim(0, 12)

            plt.savefig('./Fitness/Fitness_Progress_{}.png'.format(self.filename))
            plt.close()

            # Here we plot the trajectory of the best agent:
            self.plot_pop_list()
            self.print_best(n=1)
            print("Animation of best agent is saved")


    def plot_pop_list(self, n_knoblins=1):

        for i in range(n_knoblins):
            for trial_speed in ["slow", "fast"]:
                for init_target_direction in [-1, 1]:  # left(-1) or right(1)

                    self.setup(trial_speed=trial_speed)

                    self.target.velocity *= init_target_direction

                    self.implement_genome(genome_string=self.pop_list[i,2:])

                    self.run_and_plot()  # include reset of the neural system


    def print_best(self, n=5):
        print(">> {} best agent(s):".format(n))
        print(np.round(self.pop_list[0:n,0:4],3))

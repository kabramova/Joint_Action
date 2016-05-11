from JA_Simulator import *

class JA_Evolution(JA_Simulation):

    def __init__(self, pop_size=111):

        super(self.__class__, self).__init__(simlength=2789) # self.knoblin, self.simlength

        self.genome = self.create_genome(Knoblin=self.knoblin)

        self.pop_size = pop_size

        self.pop_list = self.__create_pop_list(pop_size)





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

    # TODO: Plot(Trajectories, Gif)


    def run_trials(self):

        fitness_per_trials = []

        for trial_speed in ["slow", "fast"]:
            for auditory_condition in [False, True]:
                for init_target_direction in [-1, 1]:  # left(-1) or right(1)

                    self.setup(trial_speed=trial_speed, auditory_condition=auditory_condition)
                    self.target.velocity *= init_target_direction

                    fitness = self.run()
                    fitness_per_trials.append(fitness)

                    self.reset_neural_system()

        fitness = np.mean(fitness_per_trials)
        print("Average fitness over all trials:", fitness)

        return fitness


    def run_population(self):

        for i, string in enumerate(self.pop_list):
            genome = string[2:]
            self.knoblin = Knoblin()
            self.implement_genome(genome_string=genome)
            # Run all trials an save fitness in pop_list:
            self.pop_list[i, 1] = self.run_trials()

        self.pop_list = copy.copy(mat_sort(self.pop_list, index=1)) # sorts the pop_list, best agents on top



    def _evolute(self, mutation_var=.25, fts=False):
        '''

        If fitness proportionate selection (fts) = False:
            +: sexual reproduction, saves best, adds new random bots
            -: Computationally expensive.

            1) Takes the best agent and copy it twice in new population.
            2) Takes the second best agent and copy it once in new population.
            3) Creates two children of two parents. Since step 1) & 2) we have a chance of genetic crossover of 100%.
                Furthermore, we use whole sections of the genome for the crossover (e.g. all W, or all Thetas)
            4) Fitness-proportionate selection of 2 further agents
            5) Fill the rest with randomly created agents

            6) All but the first best agent will fall under a mutation with a variance of .25 (default)
                - time constant: τ (tau) in range [1, 10]
                - weights: w (weights of interneurons, sensory and motor neurons) in range [-13, 13]
                - bias: θ (theta) in range [-13, 13]

        > > > > > < < < < < > > > > > < < < < < > > > > > < < < < < > > > > > < < < < < > > > > > < < < < <

        If fts = True:
            More simple, asexual, fitness-proportionate selection.

            + : Computationally more efficient.
            - : Might need more Generations to converge

             All new agents will fall under a mutation with a variance of .25 (default):
                - time constant: τ (tau) in range [1, 10]
                - weights: w (weights of interneurons, sensory and motor neurons) in range [-13, 13]
                - bias: θ (theta) in range [-13, 13]

        :param mutation_var: 0.25 by default, according to Agmon & Beer (2013)
        :return: self.pop_list = repopulated list (new_population)
        '''

        gens = self.gen_code()

        if fts == True:

            new_population = np.zeros(self.pop_list.shape)  # This will be turned in the end...

            ## Algorithm for fitness proportionate selection:
            # Source: http://stackoverflow.com/questions/298301/roulette-wheel-selection-algorithm/320788#320788
            # >>

            fitness = copy.copy(self.pop_list[:, 1])
            fitness = 1 - normalize(fitness)  # sign is correct, apparently

            total_fitness = sum(fitness)
            relative_fitness = [f / total_fitness for f in fitness]

            probs = [sum(relative_fitness[:i + 1]) for i in range(len(relative_fitness))]

            for n in range(new_population.shape[0]):
                r = np.random.random()  # random sample of continous uniform distribution [0,1)
                for (i, individual) in enumerate(self.pop_list):
                    if r <= probs[i]:
                        new_population[n, :] = individual
                        break

                        # <<

        else:  # if fts is false: Complex Evolution

            new_population = copy.copy(self.pop_list)  # This will be turned in the end...

            new_population[0, 0] = 1  # reset enumeration for first agent
            # 1)
            new_population[1, :] = copy.copy(
                self.pop_list[0, :])  # is already on first place, here we set it again on the second place
            # 2)
            new_population[2, :] = copy.copy(self.pop_list[1, :])
            # 3)
            new_population[3, :] = copy.copy(self.pop_list[0, :])
            new_population[4, :] = copy.copy(self.pop_list[0, :])

            for i in [3, 4]:  # => new_population[(3,4),:]

                ##  Alternatively, here we pick randomly 2 single genomic loci:
                # index = np.argmax(np.random.sample(self.genome.size)) + 2 -1
                # index2 = np.argmax(np.random.sample(self.genome.size)) +2 -1
                # new_population[i,index]  = copy.copy(self.pop_list[1, index])               # crossover from second parent
                # new_population[i,index2] = copy.copy(self.pop_list[1, index2])

                ## Crossover of a whole genome section of the second parent:

                choice = np.random.choice([gen for gen in gens])  # Random choice of a section in genome

                index = 0  # indexing the section in whole genome string
                for gen in gens:
                    index += gens[gen]
                    if gen == choice:
                        break
                index += 2  # leaves the number and fitness of agent out (new_population[:,(0,1)])
                new_population[i, (index - gens[choice]):index] = copy.copy(
                    self.pop_list[1, (index - gens[choice]):index])  # crossover from second parent

                # Test: self.agent.PARAMETER (depending on choice)

            # 4)
            norm_pop = normalize(np.power(self.pop_list[2:, 1], -1)) if np.any(
                self.pop_list[2:, 1] != 0) else self.pop_list[2:, 1]
            rand_pop = np.random.sample(np.size(self.pop_list[2:, 1]))
            norm_rand = norm_pop * rand_pop
            ordered = copy.copy(self.pop_list[np.argsort(-norm_rand) + 2, :])
            new_population[5, :] = ordered[0, :]
            new_population[6, :] = ordered[1, :]

            # 5)
            for i in range(new_population[7:, :].shape[0]):
                self.agent = CatchBot()  # Create new agent
                self.genome = self.create_genome()  # ... and its genome
                new_population[7 + i, 2:] = self.genome.transpose()

        # 6) Mutation (for fts=True & False):

        AGTC = sum(gens.values()) - gens["U"]  # sum of all gen-sizes, except Tau
        U = gens["U"]  # == self.agent.Tau.size

        mu, sigma = 0, np.sqrt(mutation_var)  # mean and standard deviation

        for i in range(1 - fts, new_population.shape[0]):  # if fts = False => range(1,size), else => range(0,size)

            mutation_AGTC = np.random.normal(mu, sigma, AGTC)
            mutation_U = np.random.normal(mu, sigma, U)

            AGTC_mutated = new_population[i, 2:AGTC + 2] + mutation_AGTC

            AGTC_mutated[AGTC_mutated > self.agent.W_RANGE[1]] = self.agent.W_RANGE[
                1]  # Replace values beyond the range with max.range
            AGTC_mutated[AGTC_mutated < self.agent.W_RANGE[0]] = self.agent.W_RANGE[
                0]  # ... or min.range (T_RANGE = W.RANGE =[-13, 13])

            new_population[i, 2:AGTC + 2] = AGTC_mutated

            U_mutated = new_population[i, (AGTC + 2):] + mutation_U

            U_mutated[U_mutated > self.agent.TAU_RANGE[1]] = self.agent.TAU_RANGE[
                1]  # Replace values beyond the range with max.range
            U_mutated[U_mutated < self.agent.TAU_RANGE[0]] = self.agent.TAU_RANGE[
                0]  # ... or min.range (TAU_RANGE = [1, 10])

            new_population[i, (AGTC + 2):] = U_mutated

            new_population[i, 0] = i + 1  # reset enumeration
            new_population[i, 1] = 0  # reset fitness

        self.pop_list = copy.copy(new_population)




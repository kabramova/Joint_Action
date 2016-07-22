from JA_Simulator import *
import pickle

"""
__author__  = Simon Hofmann"
__credits__ = ["Simon Hofmann", "Katja Abramova", "Willem Zuidema"]
__version__ = "1.0.1"
__date__ "2016"
__maintainer__ = "Simon Hofmann"
__email__ = "simon.hofmann@protonmail.com"
__status__ = "Development"
"""


class JA_Evolution(JA_Simulation):

    def __init__(self, auditory_condition, pop_size=55, simlength_scalar=1, symmetrical_weights=False):

        self.symmetrical_weights = symmetrical_weights

        # self.knoblin, self.simlength, self.condition
        super(self.__class__, self).__init__(auditory_condition, symmetrical_weights=self.symmetrical_weights, simlength=2789)

        self.simlength_scalar = simlength_scalar

        self.genome_l = self.create_genome(knoblin=self.knoblin_l)
        self.genome_r = self.create_genome(knoblin=self.knoblin_r)

        self.generation = 0

        self.pop_size = pop_size

        self.pop_list_l = self.__create_pop_list(pop_size, "left")
        self.pop_list_r = self.__create_pop_list(pop_size, "right")

        self.fitness_progress = []

        self.filename = ""

    def __create_pop_list(self, pop_size, side):
        """
         :param pop_size: Amount of individuals per Population
         :return: list of agents
        """

        poplist = np.zeros((pop_size, self.genome_r.size + 2))  # self.genome_r.size is self.genome_l.size

        for i in range(pop_size):
            poplist[i, 0] = i + 1                                           # enumerate the list
            # poplist[i, 1]                                                 = fitness, is initially zero
            # the current genome will be stored
            poplist[i, 2:] = self.genome_l.transpose() if side == "left" else self.genome_r.transpose()
            if side == "left":
                self.knoblin_l = Knoblin(symmetrical_weights=self.symmetrical_weights)  # Create new agent
                self.genome_l = self.create_genome(knoblin=self.knoblin_l)  # ... and its genome
            else:  # its a bit redundant, but for the readability and comprehensibility
                self.knoblin_r = Knoblin(symmetrical_weights=self.symmetrical_weights)
                self.genome_r = self.create_genome(knoblin=self.knoblin_r)

        return poplist

    @staticmethod
    def create_genome(knoblin):

        a = np.reshape(knoblin.W,      (knoblin.W.size,       1))
        g = np.reshape(knoblin.WM,     (knoblin.WM.size,      1))
        t = np.reshape(knoblin.WV,     (knoblin.WV.size,      1))
        x = np.reshape(knoblin.WA,     (knoblin.WA.size,      1))
        c = np.reshape(knoblin.Theta,  (knoblin.Theta.size,   1))
        u = np.reshape(knoblin.Tau,    (knoblin.Tau.size,     1))

        return np.concatenate((a, g, t, x, c, u))

    def implement_genome(self, genome_string, side):

        assert genome_string.size == self.genome_l.size and genome_string.size == self.genome_r.size, "Genome has not the right size"

        knoblin = self.knoblin_l if side == "left" else self.knoblin_r

        gens = self.gen_code()

        a = gens["A"]  # knoblin.W.size
        g = gens["G"]  # knoblin.WM.size
        t = gens["T"]  # knoblin.WV.size
        x = gens["X"]  # knoblin.WA.size
        c = gens["C"]  # knoblin.Theta.size
        u = gens["U"]  # knoblin.Tau.size

        w = genome_string[:a]
        wm = genome_string[a:a + g]
        wv = genome_string[a + g:a + g + t]
        wa = genome_string[a + g + t:a + g + t + x]
        theta = genome_string[a + g + t + x:a + g + t + x + c]
        tau = genome_string[a + g + t + x + c:a + g + t + x + c + u]

        # self.knoblin_l or _r (respectively) will be updated
        knoblin.W = np.reshape(w, (knoblin.N, knoblin.N))
        knoblin.WM = np.matrix(np.reshape(wm, (g, 1)))
        knoblin.WV = np.matrix(np.reshape(wv, (t, 1)))
        knoblin.WA = np.matrix(np.reshape(wa, (x, 1)))
        knoblin.Theta = np.matrix(np.reshape(theta, (c, 1)))
        knoblin.Tau = np.matrix(np.reshape(tau, (u, 1)))

        # Update the self.genome:
        if not isinstance(genome_string, np.matrix):
            genome_string = np.matrix(genome_string).transpose()

        if side == "left":
            self.genome_l = genome_string
        else:  # side == "right"
            self.genome_r = genome_string

    def run_trials(self):

        fitness_per_trials = []

        for trial_speed in ["slow", "fast"]:
            for init_target_direction in [-1, 1]:  # left(-1) or right(1)

                self.reset_neural_system()

                self.setup(trial_speed=trial_speed, simlength_scalar=self.simlength_scalar)
                self.target.velocity *= init_target_direction

                # Run trial:
                fitness = self.run()
                fitness_per_trials.append(fitness)

        fitness = np.mean(fitness_per_trials)
        # print("Average fitness over all 8 trials:", np.round(fitness,2))

        return fitness

    def _run_population(self, n_cpu, splitter=False):

        first_runs = False

        if not splitter:  # ==False
            for i in range(self.pop_size):

                string_l = copy.copy(self.pop_list_l[i, :])
                string_r = copy.copy(self.pop_list_r[i, :])

                if string_l[1] == 0.0 or string_r[1] == 0.0:  # run only if fitness is no evaluated yet
                    genome_l = string_l[2:]
                    genome_r = string_r[2:]
                    self.knoblin_l = Knoblin(symmetrical_weights=self.symmetrical_weights)
                    self.knoblin_r = Knoblin(symmetrical_weights=self.symmetrical_weights)
                    self.implement_genome(genome_string=genome_l, side="left")
                    self.implement_genome(genome_string=genome_r, side="right")

                    # Run all trials an save fitness in pop_list:
                    ticker = 10
                    # this way because it ignores the two first spots in pop_list, since they run already.
                    if i % ticker == 0 or i < 10:
                        if i % ticker == 0:
                            fill = i+ticker if i <= self.pop_size-ticker else self.pop_size
                            first_runs = True
                            print("Generation {}: Run trials for Agent Pair {}-{}".format(self.generation, i + 1, fill))
                        if i < 10 and not first_runs:
                            fill = ticker
                            first_runs = True
                            print("Generation {}: Run trials for Agent Pair {}-{} (Fitness of first agent pairs was "
                                  "already evaluated)".format(self.generation, i + 1, fill))

                    fitness = self.run_trials()
                    self.pop_list_l[i, 1] = fitness
                    self.pop_list_r[i, 1] = fitness

            self.pop_list_l = copy.copy(mat_sort(self.pop_list_l, index=1))     # sorts the pop_list, best agents on top
            self.pop_list_r = copy.copy(mat_sort(self.pop_list_r, index=1))     # sorts the pop_list, best agents on top

        # If Splitter is active:
        if not isinstance(splitter, bool):

            # n_cpu: is adaptable to the number of Processors on the server(s)

            split_size = int(self.pop_size / n_cpu)      # is self.pop_list_r.shape[0]
            rest = self.pop_size - split_size * n_cpu
            split_size = split_size + rest if splitter == 1 else split_size
            # first cpu can calculate more. This ballance out the effect, that first two Knoblins dont have to be
            # computer (if Generation > 0)

            if splitter == 1:
                start = 0
            else:
                start = split_size * (splitter - 1) + rest

            end = split_size * splitter + rest

            for i in range(start, end):
                string_l = self.pop_list_l[i, :]
                string_r = self.pop_list_r[i, :]

                if string_l[1] == 0.0 or string_r[1] == 0.0:  # run only if fitness is no evaluated yet
                    genome_l = string_l[2:]
                    genome_r = string_r[2:]
                    self.knoblin_l = Knoblin(symmetrical_weights=self.symmetrical_weights)
                    self.knoblin_r = Knoblin(symmetrical_weights=self.symmetrical_weights)
                    self.implement_genome(genome_string=genome_l, side="left")
                    self.implement_genome(genome_string=genome_r, side="right")

                    # Run all trials an save fitness in pop_list:
                    ticker = 10
                    # this way because it ignores the two first spots in pop_list, since they run already.
                    if i % ticker == 0 or i < 10:
                        if i % ticker == 0:
                            fill = i + ticker if i <= self.pop_size - ticker else self.pop_size
                            first_runs = True
                            print("Splitter{}: Generation {}: Run trials for Agent Pair {}-{}".format(splitter,
                                                                                                      self.generation,
                                                                                                      i + 1, fill))
                        if i < 10 and not first_runs:
                            fill = ticker
                            first_runs = True
                            print("Splitter{}: Generation {}: Run trials for Agent Pair {}-{} "
                                  "(Fitness of first agent pairs was already evaluated)".format(splitter,
                                                                                                self.generation,
                                                                                                i + 1, fill))

                    fitness = self.run_trials()
                    self.pop_list_l[i, 1] = fitness
                    self.pop_list_r[i, 1] = fitness

            # Save splitted Files now:
            if splitter < n_cpu:
                np.save("./temp/JA_Poplist_part_L.{}.Generation.{}.cond{}.npy".format(splitter,
                                                                                      self.generation,
                                                                                      self.condition),
                        self.pop_list_l[range(start, end), :])
                np.save("./temp/JA_Poplist_part_R.{}.Generation.{}.cond{}.npy".format(splitter,
                                                                                      self.generation,
                                                                                      self.condition),
                        self.pop_list_r[range(start, end), :])

            # Check for last splitter, whether all files are there:
            if splitter == n_cpu:  # = max number of splitters

                count = 0
                while count != n_cpu - 1:
                    count = 0

                    for n in range(1, n_cpu):
                        if os.path.isfile("./temp/JA_Poplist_part_L.{}.Generation.{}.cond{}.npy".format(n,
                                                                                                        self.generation,
                                                                                                        self.condition)) \
                                and os.path.isfile("./temp/JA_Poplist_part_R.{}.Generation.{}.cond{}.npy".format(n,
                                                                                                                 self.generation,
                                                                                                                 self.condition)):
                            count += 1
                            if count == n_cpu - 1:
                                print("All {} files of Generation {} exist".format(n_cpu - 1, self.generation))

                    time.sleep(1)  # wait 1sec before back in the loop

                # Last splitter integrates all files again:
                for save_counter in range(1, n_cpu):
                    split_size = int(self.pop_size / n_cpu)
                    split_size = split_size + rest if save_counter == 1 else split_size
                    if save_counter == 1:
                        start = 0
                    else:
                        start = split_size * (save_counter - 1) + rest

                    end = split_size * save_counter + rest

                    poplist_part_l = np.load("./temp/JA_Poplist_part_L.{}.Generation.{}.cond{}.npy".format(save_counter,
                                                                                                           self.generation,
                                                                                                           self.condition))
                    poplist_part_r = np.load("./temp/JA_Poplist_part_R.{}.Generation.{}.cond{}.npy".format(save_counter,
                                                                                                           self.generation,
                                                                                                           self.condition))

                    self.pop_list_l[range(start, end), :] = poplist_part_l  # or self.pop_list[start:end]
                    self.pop_list_r[range(start, end), :] = poplist_part_r

                print("All splitted poplist_parts successfully implemented")

                # Remove files out of dictionary
                for rm in range(1, n_cpu):
                    os.remove("./temp/JA_Poplist_part_L.{}.Generation.{}.cond{}.npy".format(rm,
                                                                                            self.generation,
                                                                                            self.condition))
                    os.remove("./temp/JA_Poplist_part_R.{}.Generation.{}.cond{}.npy".format(rm,
                                                                                            self.generation,
                                                                                            self.condition))

                if os.path.isfile("./temp/Poplist_L_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu,
                                                                                                self.generation - 1,
                                                                                                self.condition))\
                        and os.path.isfile("./temp/Poplist_R_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu,
                                                                                                         self.generation - 1,
                                                                                                         self.condition)):
                    os.remove("./temp/Poplist_L_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu,
                                                                                            self.generation - 1,
                                                                                            self.condition))
                    os.remove("./temp/Poplist_R_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu,
                                                                                            self.generation - 1,
                                                                                            self.condition))

                # sorts the pop_list, best agents on top
                self.pop_list_l = copy.copy(mat_sort(self.pop_list_l, index=1))
                self.pop_list_r = copy.copy(mat_sort(self.pop_list_r, index=1))

    def gen_code(self):

        assert self.genome_l.size == self.genome_r.size, "Genomes of left and right Agent don't have the same size!"

        gens = OrderedDict([("A", self.knoblin_l.W.size),       # could be also self.knoblin_r
                            ("G", self.knoblin_l.WM.size),
                            ("T", self.knoblin_l.WV.size),
                            ("X", self.knoblin_l.WA.size),
                            ("C", self.knoblin_l.Theta.size),
                            ("U", self.knoblin_l.Tau.size)])
        return gens

    def _reproduction(self, mutation_var):

        """
        Combination of asexual (fitness proportionate selection (fps)) and sexual reproduction
            Minimal population size = 10
            1) Takes the two best agents (couple) and copy them in new population lists.
            2) Based on pop_size, creates 2-10 children pairs (parents: two best pairs)
                - Since step 1) we have a chance of genetic crossover of 100%.
                - we use whole sections of the genome for the crossover (e.g. all W, or all Thetas)
                - 20% of population size and max. 10
            3) Fitness proportionate selection of 40% (+ 1/2 fill up)
            4) Fill with randomly created agents, 20% (+ 1/2 fill up)
            5) All but the first two best agent-pairs will fall under a mutation with a variance of .02 (default)
                - time constant: τ (tau) in range [1, 10]
                - weights: w (weights of interneurons, sensory and motor neurons) in range [-13, 13]
                - bias: θ (theta) in range [-13, 13]
            6) Shuffle half of the agents of step 3) and step4) in each list
                - this creates new pairs with former good agents
                - successful agents will have new partners.
                - The partners will be either other successful agents from other list [step 3)] or random new partners [step 4)]

        :param mutation_var: given by run_evolution()
        :return: self.pop_list = repopulated list (new_population)
        """

        gens = self.gen_code()

        new_population_l = np.zeros(self.pop_list_l.shape)
        new_population_r = np.zeros(self.pop_list_r.shape)

        # 1) Takes the two best agents and copy them in new population.
        n_parents = 2
        new_population_l[0:n_parents, :] = copy.copy(self.pop_list_l[(0, 1), :])
        new_population_r[0:n_parents, :] = copy.copy(self.pop_list_r[(0, 1), :])

        # 2) Based on pop_size, creates 2-10 children (parents: two best agents)
        n_children = int(np.round(self.pop_size*0.2) if np.round(self.pop_size*0.2) < 10 else 10)

        for n in range(n_children):
            new_population_l[2+n, 2:] = copy.copy(self.pop_list_l[0, 2:])
            new_population_r[2+n, 2:] = copy.copy(self.pop_list_r[0, 2:])

            # Crossover of a whole genome section of the second parent-pair:
            choice = np.random.choice([gen for gen in gens])  # Random choice of a section in genome

            index = 0  # indexing the section in whole genome string
            for gen in gens:
                index += gens[gen]
                if gen == choice:
                    break
            index += 2  # leaves the number and fitness of agent out (new_population[:,(0,1)])

            # crossover from second parent pair
            new_population_l[2+n, (index - gens[choice]):index] = copy.copy(self.pop_list_l[1,
                                                                            (index - gens[choice]):index])
            new_population_r[2+n, (index - gens[choice]):index] = copy.copy(self.pop_list_r[1,
                                                                            (index - gens[choice]):index])

        # 3) Fitness proportionate selection of 40% (+ 1/2 fill up)

        # Define the number of agents via fps & via random instantiation
        n_family = n_parents + n_children
        n_fps = int(np.round(self.pop_size*0.4))
        n_random = int(np.round(self.pop_size*0.2))

        if (self.pop_size - (n_family + n_fps + n_random)) != 0:
            rest = self.pop_size - (n_family + n_fps + n_random)  # rest has to be filled up

            odd = 1 if rest % 2 > 0 else 0  # if rest is odd(1) else even(0)
            n_fps += int((rest+odd)/2)
            n_random += int((rest-odd)/2)

        # Algorithm for fitness proportionate selection:
        # (Source: http://stackoverflow.com/questions/298301/roulette-wheel-selection-algorithm/320788#320788)

        assert np.all(np.array(self.pop_list_l[:, 1]) == np.array(self.pop_list_r[:, 1])), "Fitness of each partner must be the same"
        fitness = copy.copy(self.pop_list_l[:, 1])  # is self.pop_list_r[:, 1]
        fitness = 1 - normalize(fitness)  # sign is correct, apparently

        total_fitness = sum(fitness)
        relative_fitness = [f / total_fitness for f in fitness]

        probs = [sum(relative_fitness[:i + 1]) for i in range(len(relative_fitness))]

        for n in range(n_family, n_family+n_fps):   # or range(n_family, self.pop_size-n_random-1)
            r = np.random.random()        # random sample of continuous uniform distribution [0,1)
            for (i, individual_l) in enumerate(self.pop_list_l):
                individual_r = self.pop_list_r[i, :]
                if r <= probs[i]:
                    new_population_l[n, :] = individual_l
                    new_population_r[n, :] = individual_r
                    break

        # 4) Fill with randomly created agents, 20% (+ 1/2 fill up)
        n_fitfamily = n_family + n_fps
        for n in range(n_fitfamily, n_fitfamily+n_random):
            self.knoblin_l = Knoblin(symmetrical_weights=self.symmetrical_weights)  # Create random new agents
            self.knoblin_r = Knoblin(symmetrical_weights=self.symmetrical_weights)
            self.genome_l = self.create_genome(knoblin=self.knoblin_l)              # ... and their genomes
            self.genome_r = self.create_genome(knoblin=self.knoblin_r)
            new_population_l[n, 2:] = self.genome_l.transpose()
            new_population_r[n, 2:] = self.genome_r.transpose()

        # 5) All but the first two best agent pairs will fall under a mutation with a variance of .02 (default)

        agtxc = sum(gens.values()) - gens["U"]  # sum of all gen-sizes, except Tau
        u = gens["U"]  # is self.knoblin.Tau.size

        mu, sigma = 0, np.sqrt(mutation_var)  # mean and standard deviation

        # we start with the 3rd agent and end with the agents via fps, rest is random, anyways:
        for i in range(n_parents, n_fitfamily):

            mutation_agtxc_l = np.random.normal(mu, sigma, agtxc)
            mutation_agtxc_r = np.random.normal(mu, sigma, agtxc)
            mutation_u_l = np.random.normal(mu, sigma, u)
            mutation_u_r = np.random.normal(mu, sigma, u)

            agtxc_mutated_l = new_population_l[i, 2: agtxc+2] + mutation_agtxc_l
            agtxc_mutated_r = new_population_r[i, 2: agtxc+2] + mutation_agtxc_r

            # Replace values beyond the range with max.range, or min.range (T_RANGE = W.RANGE =[-13, 13])
            agtxc_mutated_l[agtxc_mutated_l > self.knoblin_l.W_RANGE[1]] = self.knoblin_l.W_RANGE[1]
            agtxc_mutated_l[agtxc_mutated_l < self.knoblin_l.W_RANGE[0]] = self.knoblin_l.W_RANGE[0]
            agtxc_mutated_r[agtxc_mutated_r > self.knoblin_r.W_RANGE[1]] = self.knoblin_r.W_RANGE[1]
            agtxc_mutated_r[agtxc_mutated_r < self.knoblin_r.W_RANGE[0]] = self.knoblin_r.W_RANGE[0]

            # For symmetrical weights the individual mutations have to be adjusted:
            if self.symmetrical_weights:
                # Motor weights:
                motor = gens["A"]  # index for motor weights
                agtxc_mutated_l[motor + 1] = agtxc_mutated_l[motor + 0]       # Outputs from Neuron 4 to Right == 6 to Left
                agtxc_mutated_l[motor + 2] = agtxc_mutated_l[motor + 0] * -1  # Outputs from Neuron 4 to Left
                agtxc_mutated_l[motor + 3] = agtxc_mutated_l[motor + 0] * -1  # Outputs from Neuron 6 to Right

                agtxc_mutated_r[motor + 1] = agtxc_mutated_l[motor + 0]
                agtxc_mutated_r[motor + 2] = agtxc_mutated_r[motor + 0] * -1
                agtxc_mutated_r[motor + 3] = agtxc_mutated_r[motor + 0] * -1

                # Visual weights
                visual = gens["A"] + gens["G"]  # index for visual weights
                agtxc_mutated_l[visual + 3] = agtxc_mutated_l[visual + 2]       # Inputs to Neuron 2,8
                agtxc_mutated_l[visual + 1] = agtxc_mutated_l[visual + 0] * -1  # Inputs to Neuron 1

                agtxc_mutated_r[visual + 3] = agtxc_mutated_r[visual + 2]
                agtxc_mutated_r[visual + 1] = agtxc_mutated_r[visual + 0] * -1

                # Auditory weights
                auditory = gens["A"] + gens["G"] + gens["T"]  # index for auditory weights
                agtxc_mutated_l[auditory + 2] = agtxc_mutated_l[auditory + 0]       # Inputs to Neuron 3,7
                agtxc_mutated_l[auditory + 3] = agtxc_mutated_l[auditory + 1] * -1  # Inputs to Neuron 5

                agtxc_mutated_r[auditory + 2] = agtxc_mutated_r[auditory + 0]
                agtxc_mutated_r[auditory + 3] = agtxc_mutated_r[auditory + 1] * -1

            # Inject mutation in new population:
            new_population_l[i, 2: agtxc+2] = agtxc_mutated_l
            new_population_r[i, 2: agtxc+2] = agtxc_mutated_r

            # Tau mutation:
            u_mutated_l = new_population_l[i, (agtxc + 2):] + mutation_u_l
            u_mutated_r = new_population_r[i, (agtxc + 2):] + mutation_u_r

            # Replace values beyond the range with max.range or min.range (TAU_RANGE = [1, 10])
            u_mutated_l[u_mutated_l > self.knoblin_l.TAU_RANGE[1]] = self.knoblin_l.TAU_RANGE[1]
            u_mutated_l[u_mutated_l < self.knoblin_l.TAU_RANGE[0]] = self.knoblin_l.TAU_RANGE[0]
            u_mutated_r[u_mutated_r > self.knoblin_r.TAU_RANGE[1]] = self.knoblin_r.TAU_RANGE[1]
            u_mutated_r[u_mutated_r < self.knoblin_r.TAU_RANGE[0]] = self.knoblin_r.TAU_RANGE[0]

            # Inject Tau-mutation in new population:
            new_population_l[i, (agtxc + 2):] = u_mutated_l
            new_population_r[i, (agtxc + 2):] = u_mutated_r

        # 6) Shuffle half of the agents of step 3) and step4) in each list
        n_half_fit_family = n_family + int(np.round(n_fps/2))
        n_half_rand_fitfamily = n_fitfamily + int(np.round(n_random/2, 2))

        new_population_l[:, 0] = range(1, self.pop_size + 1)  # enumerate list (1 to ...)
        new_population_r[:, 0] = range(1, self.pop_size + 1)

        np.random.shuffle(new_population_l[n_half_fit_family:n_half_rand_fitfamily, 0])  # shuffle the specific section
        np.random.shuffle(new_population_r[n_half_fit_family:n_half_rand_fitfamily, 0])

        new_population_l = new_population_l[np.argsort(new_population_l[:, 0])]  # sort poplist according to shuffling
        new_population_r = new_population_r[np.argsort(new_population_r[:, 0])]

        # Reset enumeration and fitness (except first two agents)
        new_population_l[:, 0] = range(1, self.pop_size+1)
        new_population_r[:, 0] = range(1, self.pop_size+1)
        new_population_l[n_parents:, 1] = 0
        new_population_r[n_parents:, 1] = 0

        self.pop_list_l = new_population_l
        self.pop_list_r = new_population_r

    def run_evolution(self, generations, mutation_var=0.10, splitter=False, n_cpu=6):
        """
        :param generations: number of generations to run
        :param mutation_var: mutation_var: 0.02 by default, turned out to be better
        :param splitter: if parallel processing
        :param n_cpu: number of parallel processes
        """

        assert isinstance(n_cpu, int) and n_cpu > 0, "n_cpu must be greater than zero (int)"

        if not splitter:  # == False
            save = save_request()
            print("No Splitter is used")
        else:
            save = True

        # Run evolution:
        if splitter == n_cpu or not splitter:
            if self.generation == 0:
                self.fitness_progress = np.zeros((generations, 6))
            else:
                self.fitness_progress = np.append(self.fitness_progress, np.zeros((generations, 6)), axis=0)

        for i in range(generations):

            if splitter == n_cpu or not splitter:
                start_timer = datetime.datetime.now().replace(microsecond=0)

            # Create new Generation
            if self.generation > 0:  # and (splitter == n_cpu or splitter is False):
                self._reproduction(mutation_var=mutation_var)

            # Evaluate fitness of each member
            self._run_population(splitter=splitter, n_cpu=n_cpu)

            # Saves Poplists for the last split
            if splitter == n_cpu:  # These files will be automatically deleted in _run_popoulation()
                np.save("./temp/Poplist_L_Splitter{}.Generation.{}.cond{}.npy".format(splitter, self.generation, self.condition), self.pop_list_l)
                np.save("./temp/Poplist_R_Splitter{}.Generation.{}.cond{}.npy".format(splitter, self.generation, self.condition), self.pop_list_r)

            # The Other scripts wait for the united poplist version from the last splitter
            if splitter < n_cpu and not isinstance(splitter, bool):
                while True:
                    if not os.path.isfile("./temp/Poplist_L_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu, self.generation, self.condition))\
                            or not os.path.isfile("./temp/Poplist_R_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu,
                                                                                                                self.generation,
                                                                                                                self.condition)):
                        time.sleep(1)
                    else:
                        break

                # Implement the united Poplists
                self.pop_list_l = np.load("./temp/Poplist_L_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu, self.generation, self.condition))
                self.pop_list_r = np.load("./temp/Poplist_R_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu, self.generation, self.condition))

            # Updated Generation counter:
            self.generation += 1

            # Saves fitness progress for the five best agents:
            if splitter == n_cpu or not splitter:
                self.fitness_progress[self.generation-1, 1:] = np.round(self.pop_list_l[0:5, 1], 2)  # is self.pop_list_r
                self.fitness_progress[self.generation-1, 0] = self.generation
                print("Generation {}: Fitness (5 best Agents): {}".format(self.generation-1, self.fitness_progress[i, 1:]))

                # Estimate Duration of Evolution
                end_timer = datetime.datetime.now().replace(microsecond=0)
                duration = end_timer - start_timer
                rest_duration = duration * (generations - (i + 1))
                print("Time passed to evolve Generation {}: {} [h:m:s]".format(self.generation-1, duration))
                print("Estimated time to evolve the rest {} Generations: {} [h:m:s]".format(generations-(i + 1), rest_duration))

        # Remove remaining temporary files out of dictionary

        if not isinstance(splitter, bool):
            # First each script has to show that it is done:
            np.save("./temp/JA_Splitter{}.DONE.cond{}.npy".format(splitter, self.condition), splitter)

        if splitter == n_cpu:  # Check whether all scripts are done
            counter = 0
            while counter != n_cpu:
                counter = 0
                for split_count in range(1, n_cpu + 1):
                    if not os.path.isfile("./temp/JA_Splitter{}.DONE.cond{}.npy".format(split_count, self.condition)):
                        time.sleep(.2)
                    else:
                        counter += 1

            for split_count in range(1, n_cpu + 1):
                os.remove("./temp/JA_Splitter{}.DONE.cond{}.npy".format(split_count, self.condition))

        # Save in external file:
        if save and (splitter == n_cpu or not splitter):
            # Add Information, if weights were held symmetrical:
            symmetry = ".sym_weights" if self.symmetrical_weights else ""

            self.filename = "Gen{}-{}.popsize{}.mut{}.sound_cond={}{}.JA.joint(Fitness{})".format(self.generation - generations + 1,
                                                                                                  self.generation,
                                                                                                  self.pop_size,
                                                                                                  mutation_var,
                                                                                                  self.condition,  # is pop_list_l[0,1]
                                                                                                  symmetry,
                                                                                                  np.round(self.pop_list_l[0, 1], 2))

            pickle.dump(self.pop_list_l, open('./poplists/joint/Poplist_L.{}'.format(self.filename), 'wb'))
            pickle.dump(self.pop_list_r, open('./poplists/joint/Poplist_R.{}'.format(self.filename), 'wb'))
            pickle.dump(np.round(self.fitness_progress, 2), open('./poplists/joint/Fitness_progress.{}'.format(self.filename), 'wb'))

            print('Evolution terminated. pop_lists saved \n'
                  '(Filename: "Poplist_...{}")'.format(self.filename))
        elif splitter < n_cpu and not isinstance(splitter, bool):
            print("Splitter {} terminated".format(splitter))
        else:
            print('Evolution terminated. \n'
                  '(Caution: pop_list is not saved in external file)')

        # Remove last Poplists out of /temp folder
        if splitter == n_cpu:
            os.remove("./temp/Poplist_L_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu,
                                                                                    self.generation - 1,
                                                                                    self.condition))
            os.remove("./temp/Poplist_R_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu,
                                                                                    self.generation - 1,
                                                                                    self.condition))

    def reimplement_population(self, filename=None, plot=False, lesion=False):

        assert filename.find("joint") != -1, "Wrong file! The file needs to be from the JOINT condition"

        if lesion:
            assert plot, "If you want to lesion a population then you need to plot it!"

        if filename is None:
            if self.filename == "":
                raise ValueError("No file to reimplement")
            else:
                print("Reimplements its own pop_list files")
        else:
            self.filename = filename

        # Reimplement: pop_list, condition, Generation
        self.pop_list_l = pickle.load(open('./poplists/joint/Poplist_L.{}'.format(self.filename), 'rb'))
        self.pop_list_r = pickle.load(open('./poplists/joint/Poplist_R.{}'.format(self.filename), 'rb'))
        self.pop_size = self.pop_list_l.shape[0]  # is self.pop_list_r.shape[0]

        assert self.filename.find("False") != -1 or self.filename.find("True") != -1, "Condition is unknown (please add to filename (if known)"

        self.condition = False if self.filename.find("False") != -1 and self.filename.find("True") == -1 else True

        self.fitness_progress = pickle.load(open('./poplists/joint/Fitness_progress.{}'.format(self.filename), 'rb'))
        self.generation = int(self.fitness_progress[-1, 0])

        print(">> ...")
        print(">> File is successfully implemented")

        if plot:

            # here we plot the fitness progress of all generation
            plt.figure()
            for i in range(1, self.fitness_progress.shape[1]):
                plt.plot(self.fitness_progress[:, i])
                plt.ylim(0, 12)

            plt.savefig('./Fitness/Fitness_Progress_{}.png'.format(self.filename))
            plt.close()

            # Here we plot the trajectory of the best agent:
            output = self.plot_pop_list(knoblin_nr=1, lesion=lesion)
            self.print_best(n=1)
            print("Animation of best agent pair is saved")

            # Output contains fitness[0], trajectories[1], keypress[2], sounds[3],
            # neural_state_L[4], neural_state_L[5], neural_input_L[6], neural_input_L[7]
            return output

    def plot_pop_list(self, knoblin_nr=1, lesion=False):

        output = []
        # count = 0

        for trial_speed in ["slow", "fast"]:
            for init_target_direction in [-1, 1]:  # left(-1) or right(1)

                self.setup(trial_speed=trial_speed, simlength_scalar=self.simlength_scalar)

                self.target.velocity *= init_target_direction

                self.implement_genome(genome_string=self.pop_list_l[knoblin_nr-1, 2:], side="left")
                self.implement_genome(genome_string=self.pop_list_r[knoblin_nr-1, 2:], side="right")

                direction = "left" if init_target_direction == - 1 else "right"
                print("Create Animation of {} trial and initial Target direction to the {}".format(trial_speed,
                                                                                                   direction))
                output.append(self.run_and_plot(lesion=lesion))      # include reset of the neural system
                # output[count] = self.run_and_plot()
                # count += 1

        print("Output contains fitness[0], trajectories[1], keypress[2], sounds[3], "
              "neural_state_L[4], neural_state_L[5], neural_input_L[6], neural_input_L[7]")
        return output

    def print_best(self, n=5):
        print(">> Left Agent(s):")
        print(self.pop_list_l[0:n, 0:3], "\n")
        print(">> Right Agent(s):")
        print(self.pop_list_r[0:n, 0:3])

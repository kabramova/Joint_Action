from JA_Simulator import *
import pickle

class JA_Evolution(JA_Simulation):

    def __init__(self, auditory_condition, pop_size=111):

        super(self.__class__, self).__init__(auditory_condition, simlength=2789) # self.knoblin, self.simlength, self.condition

        self.genome_L = self.create_genome(Knoblin=self.knoblin_L)
        self.genome_R = self.create_genome(Knoblin=self.knoblin_R)

        self.generation = 0

        self.pop_size = pop_size

        self.pop_list_L = self.__create_pop_list(pop_size, "left")
        self.pop_list_R = self.__create_pop_list(pop_size, "right")

        self.filename = ""


    def __create_pop_list(self, pop_size, side):
        '''
         :param pop_size: Amount of individuals per Population
         :return: list of agents
         '''

        poplist = np.zeros((pop_size, self.genome_R.size + 2))  # self.genome_R.size == self.genome_L.size

        for i in range(pop_size):
            poplist[i, 0] = i + 1                                           # enumerate the list
            # poplist[i, 1]                                                 = fitness, is initially zero
            poplist[i, 2:] = self.genome_L.transpose() if side=="left" else self.genome_R.transpose()  # the current genome will be stored
            if side == "left":
                self.knoblin_L = Knoblin()                                  # Create new agent
                self.genome_L = self.create_genome(Knoblin=self.knoblin_L)  # ... and its genome
            else: # its a bit redundant, but for the readability and comprehensibility
                self.knoblin_R = Knoblin()
                self.genome_R = self.create_genome(Knoblin=self.knoblin_R)

        return poplist


    def create_genome(self, Knoblin):

        A = np.reshape(Knoblin.W,      (Knoblin.W.size,       1))
        G = np.reshape(Knoblin.WM,     (Knoblin.WM.size,      1))
        T = np.reshape(Knoblin.WV,     (Knoblin.WV.size,      1))
        X = np.reshape(Knoblin.WA,     (Knoblin.WA.size,      1))
        C = np.reshape(Knoblin.Theta,  (Knoblin.Theta.size,   1))
        U = np.reshape(Knoblin.Tau,    (Knoblin.Tau.size,     1))

        return np.concatenate((A, G, T, X, C, U))


    def implement_genome(self, genome_string, side):

        assert genome_string.size == self.genome_L.size and genome_string.size == self.genome_R.size, "Genome has not the right size"

        knoblin = self.knoblin_L if side == "left" else self.knoblin_R

        A = knoblin.W.size
        G = knoblin.WM.size
        T = knoblin.WV.size
        X = knoblin.WA.size
        C = knoblin.Theta.size
        U = knoblin.Tau.size

        W       = genome_string[:A]
        WM      = genome_string[A:A + G]
        WV      = genome_string[A + G:A + G + T]
        WA      = genome_string[A + G + T:A + G + T + X]
        Theta   = genome_string[A + G + T + X:A + G + T + X + C]
        Tau     = genome_string[A + G + T + X + C:A + G + T + X + C + U]

        knoblin.W = np.reshape(W, (knoblin.N, knoblin.N))
        knoblin.WM = WM
        knoblin.WV = WV
        knoblin.WA = WA
        knoblin.Theta = Theta
        knoblin.Tau = Tau

        # Update the self.genome:
        if not isinstance(genome_string, np.matrix):
            genome_string = np.matrix(genome_string).transpose()

        if side == "left":
            self.genome_L = genome_string
        else:  # side == "right"
            self.genome_R = genome_string


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


    def _run_population(self, splitter=False):

        first_runs = False

        if splitter==False:
            for i, string_L in enumerate(self.pop_list_L):

                string_R = self.pop_list_R[i,:]

                if string_L[1] == 0 or string_R[1] == 0:  # run only if fitness is no evaluated yet
                    genome_L = string_L[2:]
                    genome_R = string_R[2:]
                    self.knoblin_L = Knoblin()
                    self.knoblin_R = Knoblin()
                    self.implement_genome(genome_string=genome_L, side="left")
                    self.implement_genome(genome_string=genome_R, side="right")

                    # Run all trials an save fitness in pop_list:
                    ticker = 10
                    if i%ticker == 0 or i < 10:  # this way because it ignores the two first spots in pop_list, since they run already.
                        if i%ticker == 0:
                            fill = i+ticker if i <= self.pop_size-ticker else self.pop_size
                            first_runs = True
                            print("Generation {}: Run trials for Agent Pair {}-{}".format(self.generation, i + 1, fill))
                        if i < 10 and first_runs == False:
                            fill = ticker
                            first_runs = True
                            print("Generation {}: Run trials for Agent Pair {}-{} (Fitness of first agent pairs was already evaluated)".format(self.generation, i + 1, fill))

                    fitness = self.run_trials()
                    self.pop_list_L[i, 1] = fitness
                    self.pop_list_R[i, 1] = fitness

            self.pop_list_L = copy.copy(mat_sort(self.pop_list_L, index=1))     # sorts the pop_list, best agents on top
            self.pop_list_R = copy.copy(mat_sort(self.pop_list_R, index=1))     # sorts the pop_list, best agents on top

        # If Splitter is active:
        if not isinstance(splitter, bool):

            n_cpu = 6  # in principal this could be adapted to the number of Processors on the server(s)

            split_size = int(self.pop_list_L.shape[0] / n_cpu)      # == self.pop_list_R.shape[0]
            rest = self.pop_list_L.shape[0] - split_size * n_cpu
            split_size = split_size + rest if splitter == 1 else split_size  # first cpu can calculate more. This ballance out the effect,
                                                                             # that first two Knoblins dont have to be computer (if Generation > 0)

            if splitter == 1:
                start = 0
            else:
                start = split_size * (splitter - 1) + rest

            end = split_size * splitter + rest

            for i in range(start, end):
                string_L = self.pop_list_L[i, :]
                string_R = self.pop_list_R[i, :]

                if string_L[1] == 0 or string_R[1] == 0:  # run only if fitness is no evaluated yet
                    genome_L = string_L[2:]
                    genome_R = string_R[2:]
                    self.knoblin_L = Knoblin()
                    self.knoblin_R = Knoblin()
                    self.implement_genome(genome_string=genome_L, side="left")
                    self.implement_genome(genome_string=genome_R, side="right")

                    # Run all trials an save fitness in pop_list:
                    ticker = 10
                    if i % ticker == 0 or i < 10:  # this way because it ignores the two first spots in pop_list, since they run already.
                        if i % ticker == 0:
                            fill = i + ticker if i <= self.pop_size - ticker else self.pop_size
                            first_runs = True
                            print("Splitter{}: Generation {}: Run trials for Agent Pair {}-{}".format(splitter, self.generation, i + 1, fill))
                        if i < 10 and first_runs == False:
                            fill = ticker
                            first_runs = True
                            print("Splitter{}: Generation {}: Run trials for Agent Pair {}-{} (Fitness of first agent pairs was already evaluated)".format(splitter, self.generation, i + 1, fill))

                    fitness = self.run_trials()
                    self.pop_list_L[i, 1] = fitness
                    self.pop_list_R[i, 1] = fitness

            # Save splitted Files now:
            if splitter < n_cpu:
                np.save("./temp/Poplist_part_L.{}.Generation.{}.cond{}.npy".format(splitter, self.generation, self.condition), self.pop_list_L[range(start, end), :])
                np.save("./temp/Poplist_part_R.{}.Generation.{}.cond{}.npy".format(splitter, self.generation, self.condition), self.pop_list_R[range(start, end), :])

            # Check for last splitter, whether all files are there:
            if splitter == n_cpu:  # = max number of splitters

                count = 0
                while count != n_cpu - 1:
                    count = 0

                    for n in range(1, n_cpu):
                        if os.path.isfile("./temp/Poplist_part_L.{}.Generation.{}.cond{}.npy".format(n, self.generation, self.condition)) \
                                and os.path.isfile("./temp/Poplist_part_R.{}.Generation.{}.cond{}.npy".format(n, self.generation, self.condition)):
                            count += 1
                            if count == n_cpu - 1:
                                print("All {} files of Generation {} exist".format(n_cpu - 1, self.generation))

                    time.sleep(1) # wait 1sec before back in the loop

                # Last splitter integrates all files again:
                for save_counter in range(1, n_cpu):
                    split_size = int(self.pop_list_L.shape[0] / n_cpu)
                    split_size = split_size + rest if save_counter == 1 else split_size
                    if save_counter == 1:
                        start = 0
                    else:
                        start = split_size * (save_counter - 1) + rest

                    end = split_size * save_counter + rest

                    poplist_part_L = np.load("./temp/Poplist_part_L.{}.Generation.{}.cond{}.npy".format(save_counter, self.generation, self.condition))
                    poplist_part_R = np.load("./temp/Poplist_part_R.{}.Generation.{}.cond{}.npy".format(save_counter, self.generation, self.condition))

                    self.pop_list_L[range(start, end), :] = poplist_part_L  # or self.pop_list[start:end]
                    self.pop_list_R[range(start, end), :] = poplist_part_R

                print("All splitted poplist_parts successfully implemented")
                # Remove files out of dictionary
                for rm in range(1, n_cpu):
                    os.remove("./temp/Poplist_part_L.{}.Generation.{}.cond{}.npy".format(rm, self.generation, self.condition))
                    os.remove("./temp/Poplist_part_R.{}.Generation.{}.cond{}.npy".format(rm, self.generation, self.condition))

                if os.path.isfile("./temp/Poplist_L_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu, self.generation - 1, self.condition))\
                        and os.path.isfile("./temp/Poplist_R_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu, self.generation - 1, self.condition)):
                    os.remove("./temp/Poplist_L_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu, self.generation - 1, self.condition))
                    os.remove("./temp/Poplist_R_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu, self.generation - 1, self.condition))

                self.pop_list_L = copy.copy(mat_sort(self.pop_list_L, index=1))  # sorts the pop_list, best agents on top
                self.pop_list_R = copy.copy(mat_sort(self.pop_list_R, index=1))


    def gen_code(self):

        assert self.genome_L.size == self.genome_R.size, "Genomes of left and right Agent don't have the same size!"

        gens = OrderedDict([("A", self.knoblin_L.W.size),       # could be also self.knoblin_R
                            ("G", self.knoblin_L.WM.size),
                            ("T", self.knoblin_L.WV.size),
                            ("X", self.knoblin_L.WA.size),
                            ("C", self.knoblin_L.Theta.size),
                            ("U", self.knoblin_L.Tau.size)])
        return gens


    def _reproduction(self, mutation_var=.02):
        '''
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

        :param mutation_var: 0.02 by default, turned out to be better.
        :return: self.pop_list = repopulated list (new_population)
        '''


        gens = self.gen_code()

        new_population_L = np.zeros(self.pop_list_L.shape)
        new_population_R = np.zeros(self.pop_list_R.shape)


        # 1) Takes the two best agents and copy them in new population.
        n_parents = 2
        new_population_L[0:n_parents,:] = copy.copy(self.pop_list_L[(0,1),:])
        new_population_R[0:n_parents,:] = copy.copy(self.pop_list_R[(0,1),:])

        # 2) Based on pop_size, creates 2-10 children (parents: two best agents)
        n_children = int(np.round(self.pop_size*0.2) if np.round(self.pop_size*0.2) < 10 else 10)

        for n in range(n_children):
            new_population_L[2+n,2:] = copy.copy(self.pop_list_L[0,2:])
            new_population_R[2+n,2:] = copy.copy(self.pop_list_R[0,2:])

            ## Crossover of a whole genome section of the second parent-pair:
            choice = np.random.choice([gen for gen in gens])  # Random choice of a section in genome

            index = 0  # indexing the section in whole genome string
            for gen in gens:
                index += gens[gen]
                if gen == choice:
                    break
            index += 2  # leaves the number and fitness of agent out (new_population[:,(0,1)])

            # crossover from second parent pair
            new_population_L[2+n, (index - gens[choice]):index] = copy.copy(self.pop_list_L[1, (index - gens[choice]):index])
            new_population_R[2+n, (index - gens[choice]):index] = copy.copy(self.pop_list_R[1, (index - gens[choice]):index])


        # 3) Fitness proportionate selection of 40% (+ 1/2 fill up)

        # Define the number of agents via fps & via random instantiation
        n_family = n_parents + n_children
        n_fps    = int(np.round(self.pop_size*0.4))
        n_random = int(np.round(self.pop_size*0.2))

        if (self.pop_size - (n_family + n_fps + n_random))!= 0:
            rest = self.pop_size - (n_family + n_fps + n_random) # rest has to be filled up

            odd = 1 if rest%2>0 else 0  # if rest is odd(1) else even(0)
            n_fps += int((rest+odd)/2)
            n_random += int((rest-odd)/2)


        # Algorithm for fitness proportionate selection:
        # (Source: http://stackoverflow.com/questions/298301/roulette-wheel-selection-algorithm/320788#320788)

        assert np.all(np.array([self.pop_list_L[:, 1] == self.pop_list_R[:, 1]]) == True), "Fitness of each partner must be the same"
        fitness = copy.copy(self.pop_list_L[:, 1])  # == self.pop_list_R[:, 1]
        fitness = 1 - normalize(fitness)  # sign is correct, apparently

        total_fitness = sum(fitness)
        relative_fitness = [f / total_fitness for f in fitness]

        probs = [sum(relative_fitness[:i + 1]) for i in range(len(relative_fitness))]

        for n in range(n_family, n_family+n_fps):   # or range(n_family, self.pop_size-n_random-1)
            r = np.random.random()        # random sample of continuous uniform distribution [0,1)
            for (i, individual_L) in enumerate(self.pop_list_L):
                individual_R = self.pop_list_R[i,:]
                if r <= probs[i]:
                    new_population_L[n, :] = individual_L
                    new_population_R[n, :] = individual_R
                    break

        # 4) Fill with randomly created agents, 20% (+ 1/2 fill up)
        n_fitfamily = n_family + n_fps
        for n in range(n_fitfamily, n_fitfamily+n_random):
            self.knoblin_L = Knoblin()                                      # Create random new agents
            self.knoblin_R = Knoblin()
            self.genome_L = self.create_genome(Knoblin = self.knoblin_L)    # ... and their genomes
            self.genome_R = self.create_genome(Knoblin = self.knoblin_R)
            new_population_L[n, 2:] = self.genome_L.transpose()
            new_population_R[n, 2:] = self.genome_R.transpose()


        # 5) All but the first two best agent pairs will fall under a mutation with a variance of .02 (default)

        AGTXC = sum(gens.values()) - gens["U"]  # sum of all gen-sizes, except Tau
        U = gens["U"]  # == self.knoblin.Tau.size

        mu, sigma = 0, np.sqrt(mutation_var)  # mean and standard deviation

        for i in range(n_parents, n_fitfamily):  # we start with the 3rd agent and end with the agents via fps, rest is random, anyways.

            mutation_AGTXC_L = np.random.normal(mu, sigma, AGTXC)
            mutation_AGTXC_R = np.random.normal(mu, sigma, AGTXC)
            mutation_U_L = np.random.normal(mu, sigma, U)
            mutation_U_R = np.random.normal(mu, sigma, U)


            AGTXC_mutated_L = new_population_L[i, 2: AGTXC+2] + mutation_AGTXC_L
            AGTXC_mutated_R = new_population_R[i, 2: AGTXC+2] + mutation_AGTXC_R

            AGTXC_mutated_L[AGTXC_mutated_L > self.knoblin_L.W_RANGE[1]] = self.knoblin_L.W_RANGE[1]  # Replace values beyond the range with max.range
            AGTXC_mutated_L[AGTXC_mutated_L < self.knoblin_L.W_RANGE[0]] = self.knoblin_L.W_RANGE[0]  # ... or min.range (T_RANGE = W.RANGE =[-13, 13])
            AGTXC_mutated_R[AGTXC_mutated_R > self.knoblin_R.W_RANGE[1]] = self.knoblin_R.W_RANGE[1]
            AGTXC_mutated_R[AGTXC_mutated_R < self.knoblin_R.W_RANGE[0]] = self.knoblin_R.W_RANGE[0]

            new_population_L[i, 2: AGTXC+2] = AGTXC_mutated_L
            new_population_R[i, 2: AGTXC+2] = AGTXC_mutated_R

            U_mutated_L = new_population_L[i, (AGTXC + 2):] + mutation_U_L
            U_mutated_R = new_population_R[i, (AGTXC + 2):] + mutation_U_R

            U_mutated_L[U_mutated_L > self.knoblin_L.TAU_RANGE[1]] = self.knoblin_L.TAU_RANGE[1]  # Replace values beyond the range with max.range
            U_mutated_L[U_mutated_L < self.knoblin_L.TAU_RANGE[0]] = self.knoblin_L.TAU_RANGE[0]  # ... or min.range (TAU_RANGE = [1, 10])
            U_mutated_R[U_mutated_R > self.knoblin_R.TAU_RANGE[1]] = self.knoblin_R.TAU_RANGE[1]  # Replace values beyond the range with max.range
            U_mutated_R[U_mutated_R < self.knoblin_R.TAU_RANGE[0]] = self.knoblin_R.TAU_RANGE[0]  # ... or min.range (TAU_RANGE = [1, 10])

            new_population_L[i, (AGTXC + 2):] = U_mutated_L
            new_population_R[i, (AGTXC + 2):] = U_mutated_R


        # 6) Shuffle half of the agents of step 3) and step4) in each list
        n_half_fit_family = n_family + int(np.round(n_fps/2))
        n_half_rand_fitfamily =  n_fitfamily + int(np.round(n_random/2,2))

        new_population_L[:, 0] = range(1, self.pop_size + 1) # enumerate list (1 to ...)
        new_population_R[:, 0] = range(1, self.pop_size + 1)

        np.random.shuffle(new_population_L[n_half_fit_family:n_half_rand_fitfamily, 0]) # shuffle the specific section
        np.random.shuffle(new_population_R[n_half_fit_family:n_half_rand_fitfamily, 0])

        new_population_L = new_population_L[np.argsort(new_population_L[:, 0])]   # sort poplist according to shuffling
        new_population_R = new_population_R[np.argsort(new_population_R[:, 0])]


        # Reset enumeration and fitness (except first two agents)
        new_population_L[:, 0] = range(1, self.pop_size+1)
        new_population_R[:, 0] = range(1, self.pop_size+1)
        new_population_L[n_parents:, 1] = 0
        new_population_R[n_parents:, 1] = 0

        self.pop_list_L = new_population_L
        self.pop_list_R = new_population_R


    def run_evolution(self, generations, mutation_var=.02, splitter=False):

        if splitter == False:
            save = save_request()
            print("No Splitter is used")
        else:
            save = True

        n_cpu = 6

        # Run evolution:
        if splitter == n_cpu or splitter == False:
            Fitness_progress = np.zeros((generations, 6))


        for i in range(generations):

            if splitter == n_cpu or splitter == False:
                start_timer = datetime.datetime.now().replace(microsecond=0)

            # Create new Generation
            if i != 0: # and (splitter == n_cpu or splitter == False):
                self._reproduction(mutation_var)

            # Evaluate fitness of each member
            self._run_population(splitter=splitter)

            # Saves Poplists for the last split
            if splitter == n_cpu: # These files will be automatically deleted in _run_popoulation()
                np.save("./temp/Poplist_L_Splitter{}.Generation.{}.cond{}.npy".format(splitter, self.generation, self.condition), self.pop_list_L)
                np.save("./temp/Poplist_R_Splitter{}.Generation.{}.cond{}.npy".format(splitter, self.generation, self.condition), self.pop_list_R)

            # The Other scripts wait for the united poplist version from the last splitter
            if splitter < n_cpu and not isinstance(splitter, bool):
                while True:
                    if not os.path.isfile("./temp/Poplist_L_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu, self.generation, self.condition))\
                            or not os.path.isfile("./temp/Poplist_R_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu, self.generation, self.condition)):
                        time.sleep(1)
                    else:
                        break

                # Implement the united Poplists
                self.pop_list_L = np.load("./temp/Poplist_L_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu, self.generation, self.condition))
                self.pop_list_R = np.load("./temp/Poplist_R_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu, self.generation, self.condition))

            # Updated Generation counter:
            self.generation += 1

            # Saves fitness progress for the five best agents:
            if splitter == n_cpu or splitter == False:
                Fitness_progress[i, 1:] = np.round(self.pop_list_L[0:5, 1], 2)  # == self.pop_list_R
                Fitness_progress[i, 0] = self.generation
                print("Generation {}: Fitness (5 best Agents): {}".format(self.generation, Fitness_progress[i, 1:]))

                # Estimate Duration of Evolution
                end_timer = datetime.datetime.now().replace(microsecond=0)
                duration = end_timer - start_timer
                rest_duration = duration * (generations - (i + 1))
                print("Time passed to evolve Generation {}: {} [h:m:s]".format(i, duration))
                print("Estimated time to evolve the rest {} Generations: {} [h:m:s]".format(generations-(i + 1), rest_duration))

        # Remove remaining temporary files out of dictionary

        np.save("./temp/JA_Splitter{}.DONE.cond{}.npy".format(splitter, self.condition), splitter)  # First each script has to show that it is done

        if splitter == n_cpu:  # Check whether all scripts are done
            counter = 0
            while counter != n_cpu:
                counter = 0
                for split_count in range(1, n_cpu + 1):
                    if not os.path.isfile("./temp/JA_Splitter{}.DONE.cond{}.npy".format(split_count, self.condition)):
                        time.sleep(.2)
                    else:
                        counter += 1

            # Remove
            os.remove("./temp/Poplist_L_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu, self.generation - 1, self.condition))
            os.remove("./temp/Poplist_R_Splitter{}.Generation.{}.cond{}.npy".format(n_cpu, self.generation - 1, self.condition))

            for split_count in range(1, n_cpu + 1):
                os.remove("./temp/JA_Splitter{}.DONE.cond{}.npy".format(split_count, self.condition))



        # Save in external file:
        if save and (splitter == n_cpu or splitter == False):
            self.filename = "Gen{}-{}.popsize{}.mut{}.sound_cond={}.JA.joint(Fitness{})".format(self.generation - generations + 1,
                                                                                                 self.generation,
                                                                                                 self.pop_size,
                                                                                                 mutation_var,
                                                                                                 self.condition,
                                                                                                 np.round(self.pop_list_L[0,1],2)) # == pop_list_L[0,1]

            pickle.dump(self.pop_list_L, open('./poplists/Joint/Poplist_L.{}'.format(self.filename), 'wb'))
            pickle.dump(self.pop_list_R, open('./poplists/Joint/Poplist_R.{}'.format(self.filename), 'wb'))
            pickle.dump(np.round(Fitness_progress, 2), open('./poplists/Joint/Fitness_progress.{}'.format(self.filename), 'wb'))

            print('Evolution terminated. pop_lists saved \n'
                  '(Filename: "Poplist_...{}")'.format(self.filename))
        elif splitter < n_cpu and not isinstance(splitter, bool):
            print("Splitter {} terminated".format(splitter))
        else:
            print('Evolution terminated. \n'
                  '(Caution: pop_list is not saved in external file)')


    def reimplement_population(self, filename=None, Plot=False):

        assert filename.find("joint") != -1, "Wrong file! The file needs to be from the JOINT condition"

        if filename is None:
            if self.filename == "":
                raise ValueError("No file to reimplement")
            else:
                print("Reimplements its own pop_list files")
        else:
            self.filename = filename

        # Reimplement: pop_list, condition, Generation
        self.pop_list_L = pickle.load(open('./poplists/Joint/Poplist_L.{}'.format(self.filename), 'rb'))
        self.pop_list_R = pickle.load(open('./poplists/Joint/Poplist_R.{}'.format(self.filename), 'rb'))
        self.pop_size = self.pop_list_L.shape[0] # == self.pop_list_R.shape[0]

        assert self.filename.find("False") != -1 or self.filename.find("True") != -1, "Condition is unknown (please add to filename (if known)"
        self.condition = False if self.filename.find("False") != -1 and self.filename.find("True") == -1 else True

        fitness_progress = pickle.load(open('./poplists/Joint/Fitness_progress.{}'.format(self.filename), 'rb'))
        self.generation = int(fitness_progress[-1, 0])

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
            print("Animation of best agent pair is saved")


    def plot_pop_list(self, n_knoblins=1):

        for i in range(n_knoblins):
            for trial_speed in ["slow", "fast"]:
                for init_target_direction in [-1, 1]:  # left(-1) or right(1)

                    self.setup(trial_speed=trial_speed)

                    self.target.velocity *= init_target_direction

                    self.implement_genome(genome_string=self.pop_list_L[i,2:], side="left")
                    self.implement_genome(genome_string=self.pop_list_R[i,2:], side="right")

                    direction = "left" if init_target_direction == - 1 else "right"
                    print("Create Animation of {} trial and initial Target direction to the {}".format(trial_speed ,direction))
                    self.run_and_plot()  # include reset of the neural system


    def print_best(self, n=5):
        print(">> Left Agent(s):")
        print(self.pop_list_L[0:n,0:3], "\n")
        print(">> Right Agent(s):")
        print(self.pop_list_R[0:n,0:3])


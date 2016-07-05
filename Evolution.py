from Simulator import *
import pickle

# Agmon & Beer (2013): "real-valued GA":
'''
"Each genetic string is a search vector of real numbers in the range 61,
and is scaled by each parameter’s defined range
(...)
The top-performing individual is copied twice into the next generation’s
population, and the rest of the population is repopulated through
fitness-proportionate selection and mutation, with a
mutation variance of 0.25." - Agmon,Beer (2013)

parameters to evolve:
- time constant: τ (tau)
- weights: w (weights of interneurons, sensory and motor neurons)
- bias: θ (theta)
'''


class Evolution(Simulate):

    def __init__(self, pop_size=10, simlength=1000):
        """
        :param pop_size:
        """
        super(self.__class__, self).__init__(simlength)  # self.agent, self.simlength

        self.genome = self.create_genome()  # vector of parameters

        self.pop_list = self.__create_pop_list(pop_size)

        self.Generation = 0

        self.filename = ""   # starts with "sim...."

    def create_genome(self):
        """
        Reshape parameter matrices into 1-D vectors and concatenate them
        :rtype: vector
        :return: vector of all parameters
        """
        a = np.reshape(self.agent.W,     (self.agent.W.size, 1))
        g = np.reshape(self.agent.WM,    (self.agent.WM.size, 1))
        t = np.reshape(self.agent.WV,    (self.agent.WV.size, 1))
        c = np.reshape(self.agent.Theta, (self.agent.Theta.size, 1))
        u = np.reshape(self.agent.Tau,   (self.agent.Tau.size, 1))

        return np.concatenate((a, g, t, c, u))

    def implement_genome(self, genome_string):

        assert genome_string.size == self.genome.size, "Genome has not the right size"

        a = self.agent.W.size
        g = self.agent.WM.size
        t = self.agent.WV.size
        c = self.agent.Theta.size
        u = self.agent.Tau.size

        w = genome_string[:a]
        wm = genome_string[a:a+g]
        wv = genome_string[a+g:a+g+t]
        theta = genome_string[a+g+t:a+g+t+c]
        tau = genome_string[a+g+t+c:a+g+t+c+u]

        self.agent.W = np.matrix(np.reshape(w,          (self.agent.N, self.agent.N)))
        self.agent.WM = np.matrix(np.reshape(wm,        (g, 1)))    # for poplists before 1.June take the reshape out (see github, also CTRNN.py)
        self.agent.WV = np.matrix(np.reshape(wv,        (t, 1)))
        self.agent.Theta = np.matrix(np.reshape(theta,  (c, 1)))
        self.agent.Tau = np.matrix(np.reshape(tau,      (u, 1)))

        # Update the self.genome:
        if not isinstance(genome_string, np.matrix):
            genome_string = np.matrix(genome_string).transpose()

        self.genome = genome_string

    def fitness(self):
        """
        Fitness is the distance to target after the simulation run.
        :rtype: distance to Target (int)
        """
        return np.linalg.norm(self.agent.position_target - self.agent.position)

    def __create_pop_list(self, pop_size):
        """
        :param pop_size: Amount of individuals per Population
        :return: ordered list (via fitness) of all agents
        """

        poplist = np.zeros((pop_size, np.size(self.genome)+2))

        for i in range(pop_size):
            poplist[i, 0] = i+1                         # enumerate the list
            poplist[i, 2:] = self.genome.transpose()    # the corresponding genome will be stored
            self.agent = CatchBot()                     # Create new agent
            self.genome = self.create_genome()          # ... and its genome

        return poplist

    def pick_best(self):
        return self.pop_list[(0, 1), :]

    def gen_code(self):
        gens = OrderedDict([("A", self.agent.W.size),
                            ("G", self.agent.WM.size),
                            ("T", self.agent.WV.size),
                            ("C", self.agent.Theta.size),
                            ("U", self.agent.Tau.size)])
        return gens

    def _reproduction(self, mutation_var, fps=False):
        """
        If fitness proportionate selection (fps) = False:
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

        If fps = True:
            More simple, asexual, fitness-proportionate selection.

            + : Computationally more efficient.
            - : Might need more Generations to converge

             All new agents will fall under a mutation with a variance of .25 (default):
                - time constant: τ (tau) in range [1, 10]
                - weights: w (weights of interneurons, sensory and motor neurons) in range [-13, 13]
                - bias: θ (theta) in range [-13, 13]

        :param mutation_var: given by run_evolution() (0.25 by default, according to Agmon & Beer (2013))
        :return: self.pop_list = repopulated list (new_population)
        """

        gens = self.gen_code()

        if fps:

            new_population = np.zeros(self.pop_list.shape)  # This will be turned in the end...

            # Algorithm for fitness proportionate selection:
            # Source: http://stackoverflow.com/questions/298301/roulette-wheel-selection-algorithm/320788#320788
            # >>

            fitness = copy.copy(self.pop_list[:, 1])
            fitness = 1-normalize(fitness)                  # sign is correct, apparently

            total_fitness = sum(fitness)
            relative_fitness = [f/total_fitness for f in fitness]

            probs = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]

            for num in range(new_population.shape[0]):
                r = np.random.random()   # random sample of continous uniform distribution [0,1)
                for (i, individual) in enumerate(self.pop_list):
                    if r <= probs[i]:
                        new_population[num, :] = individual
                        break

            # <<

        else:  # if fps is false: Complex Evolution

            new_population = copy.copy(self.pop_list)               # This will be turned in the end...

            new_population[0, 0] = 1                                # reset enumeration for first agent

            # 1)
            # is already on first place, here we set it again on the second place
            new_population[1, :] = copy.copy(self.pop_list[0, :])
            # 2)
            new_population[2, :] = copy.copy(self.pop_list[1, :])
            # 3)
            new_population[3, :] = copy.copy(self.pop_list[0, :])
            new_population[4, :] = copy.copy(self.pop_list[0, :])

            for i in [3, 4]:     # => new_population[(3,4),:]

                #  Alternatively, here we pick randomly 2 single genomic loci:
                # index = np.argmax(np.random.sample(self.genome.size)) + 2 -1
                # index2 = np.argmax(np.random.sample(self.genome.size)) +2 -1
                # new_population[i, index]  = copy.copy(self.pop_list[1, index])          # crossover from second parent
                # new_population[i, index2] = copy.copy(self.pop_list[1, index2])

                # Crossover of a whole genome section of the second parent:

                choice = np.random.choice([gen for gen in gens])  # Random choice of a section in genome

                index = 0  # indexing the section in whole genome string
                for gen in gens:
                    index += gens[gen]
                    if gen == choice:
                        break
                index += 2   # leaves the number and fitness of agent out (new_population[:,(0,1)])

                # crossover from second parent
                new_population[i, (index-gens[choice]):index] = copy.copy(self.pop_list[1, (index-gens[choice]):index])

                # Test: self.agent.PARAMETER (depending on choice)

            # 4)
            norm_pop = normalize(np.power(self.pop_list[2:, 1], -1)) if \
                np.any(self.pop_list[2:, 1] != 0) else self.pop_list[2:, 1]
            rand_pop = np.random.sample(np.size(self.pop_list[2:, 1]))
            norm_rand = norm_pop * rand_pop
            ordered = copy.copy(self.pop_list[np.argsort(-norm_rand)+2, :])
            new_population[5, :] = ordered[0, :]
            new_population[6, :] = ordered[1, :]

            # 5)
            for i in range(new_population[7:, :].shape[0]):
                self.agent = CatchBot()                         # Create new agent
                self.genome = self.create_genome()              # ... and its genome
                new_population[7+i, 2:] = self.genome.transpose()

        # 6) Mutation (for fps=True & False):

        agtc = sum(gens.values()) - gens["U"]   # sum of all gen-sizes, except Tau
        u = gens["U"]                           # == self.agent.Tau.size

        mu, sigma = 0, np.sqrt(mutation_var)    # mean and standard deviation

        for i in range(1-fps, new_population.shape[0]):  # if fps = False => range(1,size), else => range(0,size)

            mutation_agtc = np.random.normal(mu, sigma, agtc)
            mutation_u = np.random.normal(mu, sigma, u)

            agtc_mutated = new_population[i, 2:agtc+2] + mutation_agtc

            # Replace values beyond the range with max.range
            agtc_mutated[agtc_mutated > self.agent.W_RANGE[1]] = self.agent.W_RANGE[1]
            # ... or min.range (T_RANGE = W.RANGE =[-13, 13])
            agtc_mutated[agtc_mutated < self.agent.W_RANGE[0]] = self.agent.W_RANGE[0]

            new_population[i, 2:agtc+2] = agtc_mutated

            u_mutated = new_population[i, (agtc+2):] + mutation_u

            # Replace values beyond the range with max.range
            u_mutated[u_mutated > self.agent.TAU_RANGE[1]] = self.agent.TAU_RANGE[1]
            # ... or min.range (TAU_RANGE = [1, 10])
            u_mutated[u_mutated < self.agent.TAU_RANGE[0]] = self.agent.TAU_RANGE[0]

            new_population[i, (agtc+2):] = u_mutated

            new_population[i, 0] = i+1   # reset enumeration
            new_population[i, 1] = 0     # reset fitness

        self.pop_list = copy.copy(new_population)

    @staticmethod
    def _set_target(position_agent=[50, 50], angle_to_target=np.pi/2, distance=30, iscomplex=False):

        if not iscomplex:  # We just create one target, depending on the angle:
            pos_target = np.array(position_agent) + np.array([np.cos(angle_to_target), np.sin(angle_to_target)]) * distance

            return list([pos_target])  # This form of output is necessarry for _simulate_next_population()

        else:  # We create different Targets around the Agent, depending on its Position (ignoring the input angle):
            circle = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2,  7*np.pi/4]
            pos_target = []
            scalar = [.5, 2, 1, 1, 1.5, .5, 2, 1.5]

            for j, cle in enumerate(circle):
                tpos = np.array(position_agent) + np.array([np.cos(cle), np.sin(cle)]) * distance * scalar[j]
                pos_target.append(tpos)

            return pos_target

    def _simulate_next_population(self, position_agent, pos_target):
        """
        Run simulation => fitness
        We save the distance to (each) target. The fitness will be the (average) distance
        If we have more than one target:
            - each agent will run through all trials (each trial the target is on a different position).
            - we take average Fitness over all  ('complex trials')

        :param position_agent:
        :param pos_target:
        :return: Updates sorted pop_list
        """

        assert self.pop_list[-1, 1] == 0, "This population run already its simulation"

        for i, string in enumerate(self.pop_list):  # Run simulation with each agent

            genome_string = string[2:]

            fitness = []

            for tpos in pos_target:

                # reset self.agent and set new target position
                self.agent = CatchBot(position_agent=position_agent, position_target=[tpos[0], tpos[1]])

                self.implement_genome(genome_string)  # implement the current genome in agent

                self.agent.movement(self.simlength)

                fitness.append(self.fitness())

            self.pop_list[i, 1] = np.sum(fitness)/len(fitness)  # agent's average fitness will be stored

        self.pop_list = copy.copy(mat_sort(self.pop_list, index=1))

    def run_evolution(self, generations, mutation_var=0.10, complex_trials=True, fit_prop_sel=False,
                      position_agent=[50, 50], angle_to_target=np.pi/2, distance_to_target=30):
        """
        Run evolution for n-generations with particular mutation rate.

        :param generations: number of generations to run
        :param mutation_var: test out smaller value, 0.25 by default, according to Agmon & Beer (2013)
        :param complex_trials: if true multiple targets to catch
        :param fit_prop_sel: fitness proportionate selection
        :param position_agent: start position of agent in all trials
        :param angle_to_target: defines angle to target (in case of complex_trials, redundant)
        :param distance_to_target: defines corresponding distance to target (in case of complex_trials, redundant)
        """

        # Ask whether results should be saved in external file
        save = save_request()

        # Run evolution:
        fitness_progress = np.zeros((generations, 3))

        pos_target = self._set_target(position_agent=position_agent,
                                      angle_to_target=angle_to_target,
                                      distance=distance_to_target,
                                      iscomplex=complex_trials)

        for i in range(generations):

            start_timer = datetime.datetime.now().replace(microsecond=0)

            self._reproduction(mutation_var=mutation_var, fps=fit_prop_sel)

            self._simulate_next_population(position_agent=position_agent,
                                           pos_target=pos_target)

            fitness_progress[i, 1:] = np.round(self.pick_best()[:, 1], 2)

            self.Generation += 1

            fitness_progress[i, 0] = self.Generation

            print(fitness_progress[i, 1:], "Generation", self.Generation)

            # Estimate Duration of Evolution
            end_timer = datetime.datetime.now().replace(microsecond=0)
            duration = end_timer - start_timer
            rest_duration = duration * (generations - (i + 1))
            print("Time passed to evolve Generation {}: {} [h:m:s]".format(i, duration))
            print("Estimated time to evolve the rest {} Generations: {} [h:m:s]".format(generations-(i + 1),
                                                                                        rest_duration))

        # Save in external file:
        if save:

            self.filename = "sim{}.mut{}.Gen{}-{}(Fitness {})".format(self.simlength,
                                                                      mutation_var,
                                                                      self.Generation - generations + 1,
                                                                      self.Generation, np.round(self.pop_list[0, 1], 2))

            pickle.dump(self.pop_list, open('./poplists/Poplist.{}'.format(self.filename), 'wb'))
            pickle.dump(np.round(fitness_progress, 2),
                        open('./poplists/Fitness_progress.{}'.format(self.filename), 'wb'))

            print('Evolution terminated. pop_list saved \n'
                  '(Filename: "Poplist.{}")'.format(self.filename))
        else:
            print('Evolution terminated. \n'
                  '(Caution: pop_list is not saved in external file)')

    def reimplement_population(self, filename=None, plot=False):

        if filename is None:
            filename = self.filename
            print("Reimplements its own pop_list file")

        # Reimplement: pop_list, simlength, Generation
        self.pop_list = pickle.load(open('./poplists/Poplist.{}'.format(filename), 'rb'))

        self.simlength = int(filename[filename.find('m')+1: filename.find('.')])  # depends on filename

        fitness_progress = pickle.load(open('./poplists/Fitness_progress.{}'.format(filename), 'rb'))
        self.Generation = int(fitness_progress[-1, 0])

        self.filename = filename

        if plot:

            animation = animation_request()

            # here we plot the fitness progress of all generation
            plt.figure()
            plt.plot(fitness_progress[:, 1])
            plt.plot(fitness_progress[:, 2])

            # Here we plot the trajectory of the best agent:

            self.plot_pop_list(animation=animation)
            print("Plot the best agent")

            global n   # this is needed for self.close()
            n = 2

    def plot_pop_list(self, n_agents=1, position_agent=[50, 50], animation=False):

        global n
        n = n_agents

        pos_target = self._set_target(position_agent=position_agent, iscomplex=True)
        col = ["royalblue", "tomato", "palegreen", "fuchsia", "gold", "darkviolet", "darkslategray", "orange"]  # colors.cnames

        for i in range(n_agents):

            col_count = 0

            if not animation:
                plt.figure(figsize=(10, 6), dpi=80)
            else:
                plt.figure(figsize=(10, 6), dpi=40)

            # Define boarders
            plt.xlim(0, 100)
            plt.ylim(-15, 100)

            for tpos in pos_target:
                self.agent = CatchBot(position_agent=position_agent)
                self.agent.position_target = tpos
                self.implement_genome(self.pop_list[i, 2:])
                plt.plot(tpos[0], tpos[1], 's', c=col[col_count])  # Plot Targets
                self.run_and_plot(colour=col[col_count], animation=animation)  # Plot Trajectory

                col_count += 1

            plt.plot(position_agent[0], position_agent[1], 'bo')

        print(np.round(self.pop_list[0:n_agents, 0:3], 2))
        if n_agents > 1:
            print("Close all Windows with close()")

    @staticmethod
    def close():
        for j in range(n):  # n is from the global variable of plot_pop_list()/reimplement_population()
            plt.close()

# t3 = Evolution(simlength=50)
# t3.run_evolution(Generations=10)
# t3.plot_pop_list(2)
# t3.close()

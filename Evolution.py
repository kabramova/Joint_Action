from Simulator import *
from collections import OrderedDict
import pickle

## Agmon & Beer (2013): "real-valued GA":
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

    def __init__(self, pop_size = 10, simlength = 1000):
        """
        :param pop_size:
        :param args:
        """
        super(self.__class__, self).__init__(simlength)  # self.agent, self.simlength

        self.genome = self.create_genome() # vector of parameters

        self.pop_list = self.__create_pop_list(pop_size)

        self.Generation = 0

        self.filename = ""   # starts with "sim...."


    def create_genome(self):
        """
        Reshape parameter matrices into 1-D vectors and concatenate them
        :rtype: vector
        :return: vector of all parameters
        """
        A = np.reshape(self.agent.W, (self.agent.W.size,1))
        G = np.reshape(self.agent.WM,(self.agent.WM.size,1))
        T = np.reshape(self.agent.WV,(self.agent.WV.size,1))
        C = np.reshape(self.agent.Theta,(self.agent.Theta.size,1))
        U = np.reshape(self.agent.Tau,(self.agent.Tau.size,1))

        return np.concatenate((A,G,T,C,U))


    def implement_genome(self, genome_string):

        assert genome_string.size == self.genome.size, "Genome has not the right size"

        A = self.agent.W.size
        G = self.agent.WM.size
        T = self.agent.WV.size
        C = self.agent.Theta.size
        U = self.agent.Tau.size

        W, WM, WV, Theta, Tau = genome_string[:A], genome_string[A:A+G], genome_string[A+G:A+G+T], genome_string[A+G+T:A+G+T+C], genome_string[A+G+T+C:A+G+T+C+U]

        self.agent.W      = np.reshape(W, (self.agent.N, self.agent.N))
        self.agent.WM     = WM
        self.agent.WV     = WV
        self.agent.Theta  = Theta
        self.agent.Tau    = Tau

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
        '''

        :param pop_size: Amount of individuals per Population
        :return: ordered list (via fitness) of all agents
        '''

        poplist = np.zeros((pop_size, np.size(self.genome)+2))

        for i in range(pop_size):
            poplist[i,0] = i+1                         # enumerate the list
            poplist[i,2:] = self.genome.transpose()    # the corresponding genome will be stored
            self.agent = CatchBot()                    # Create new agent
            self.genome = self.create_genome()         # ... and its genome

        return poplist


    def pick_best(self):
        return self.pop_list[(0,1),:]


    def gen_code(self):
        gens = OrderedDict([("A", self.agent.W.size),
                            ("G", self.agent.WM.size),
                            ("T", self.agent.WV.size),
                            ("C", self.agent.Theta.size),
                            ("U", self.agent.Tau.size)])
        return gens


    def _reproduction(self, mutation_var = .25, fps = False):
        '''

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

        :param mutation_var: 0.25 by default, according to Agmon & Beer (2013)
        :return: self.pop_list = repopulated list (new_population)
        '''

        gens = self.gen_code()

        if fps == True:

            new_population = np.zeros(self.pop_list.shape)  # This will be turned in the end...

            ## Algorithm for fitness proportionate selection:
            # Source: http://stackoverflow.com/questions/298301/roulette-wheel-selection-algorithm/320788#320788
            # >>

            fitness = copy.copy(self.pop_list[:,1])
            fitness = 1-normalize(fitness)                  # sign is correct, apparently

            total_fitness = sum(fitness)
            relative_fitness = [f/total_fitness for f in fitness]

            probs = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]

            for n in range(new_population.shape[0]):
                r = np.random.random()   # random sample of continous uniform distribution [0,1)
                for (i, individual) in enumerate(self.pop_list):
                    if r <= probs[i]:
                        new_population[n,:] = individual
                        break

            # <<

        else: # if fps is false: Complex Evolution

            new_population = copy.copy(self.pop_list)               # This will be turned in the end...

            new_population[0,0] = 1                                 # reset enumeration for first agent
            # 1)
            new_population[1,:] = copy.copy(self.pop_list[0,:])     # is already on first place, here we set it again on the second place
            # 2)
            new_population[2,:] = copy.copy(self.pop_list[1,:])
            # 3)
            new_population[3,:] = copy.copy(self.pop_list[0,:])
            new_population[4,:] = copy.copy(self.pop_list[0,:])


            for i in [3,4]:     # => new_population[(3,4),:]

                ##  Alternatively, here we pick randomly 2 single genomic loci:
                # index = np.argmax(np.random.sample(self.genome.size)) + 2 -1
                # index2 = np.argmax(np.random.sample(self.genome.size)) +2 -1
                # new_population[i,index]  = copy.copy(self.pop_list[1, index])               # crossover from second parent
                # new_population[i,index2] = copy.copy(self.pop_list[1, index2])

                ## Crossover of a whole genome section of the second parent:

                choice = np.random.choice([gen for gen in gens]) # Random choice of a section in genome

                index = 0 # indexing the section in whole genome string
                for gen in gens:
                    index += gens[gen]
                    if gen == choice:
                        break
                index += 2   # leaves the number and fitness of agent out (new_population[:,(0,1)])
                new_population[i,(index-gens[choice]):index] = copy.copy(self.pop_list[1,(index-gens[choice]):index]) # crossover from second parent

                # Test: self.agent.PARAMETER (depending on choice)


            # 4)
            norm_pop = normalize( np.power(self.pop_list[2:,1], -1 ) ) if np.any(self.pop_list[2:,1] != 0) else self.pop_list[2:,1]
            rand_pop = np.random.sample(np.size(self.pop_list[2:,1]))
            norm_rand = norm_pop * rand_pop
            ordered = copy.copy(self.pop_list[np.argsort(-norm_rand)+2,:])
            new_population[5,:] = ordered[0,:]
            new_population[6,:] = ordered[1,:]

            # 5)
            for i in range(new_population[7:,:].shape[0]):
                self.agent = CatchBot()                         # Create new agent
                self.genome = self.create_genome()              # ... and its genome
                new_population[7+i,2:] = self.genome.transpose()


        # 6) Mutation (for fps=True & False):

        AGTC = sum(gens.values()) - gens["U"]   # sum of all gen-sizes, except Tau
        U    = gens["U"]                        # == self.agent.Tau.size

        mu, sigma = 0, np.sqrt(mutation_var)    # mean and standard deviation

        for i in range(1-fps, new_population.shape[0]):  # if fps = False => range(1,size), else => range(0,size)

            mutation_AGTC = np.random.normal(mu, sigma, AGTC)
            mutation_U    = np.random.normal(mu, sigma, U)


            AGTC_mutated =  new_population[i,2:AGTC+2] + mutation_AGTC

            AGTC_mutated[AGTC_mutated > self.agent.W_RANGE[1]] = self.agent.W_RANGE[1]       # Replace values beyond the range with max.range
            AGTC_mutated[AGTC_mutated < self.agent.W_RANGE[0]] = self.agent.W_RANGE[0]       # ... or min.range (T_RANGE = W.RANGE =[-13, 13])

            new_population[i,2:AGTC+2] = AGTC_mutated


            U_mutated = new_population[i,(AGTC+2):] + mutation_U

            U_mutated[U_mutated > self.agent.TAU_RANGE[1]] = self.agent.TAU_RANGE[1]        # Replace values beyond the range with max.range
            U_mutated[U_mutated < self.agent.TAU_RANGE[0]] = self.agent.TAU_RANGE[0]        # ... or min.range (TAU_RANGE = [1, 10])

            new_population[i,(AGTC+2):] = U_mutated


            new_population[i,0] = i+1   # reset enumeration
            new_population[i,1] = 0     # reset fitness


        self.pop_list = copy.copy(new_population)


    def _set_target(self, position_agent=[50,50], angle_to_target = np.pi/2 , distance = 30, complex = False):

        if not complex: # We just create one target, depending on the angle:
            pos_target = np.array(position_agent) + np.array([np.cos(angle_to_target), np.sin(angle_to_target)]) * distance

            return list([pos_target]) # This form of output is necessarry for _simulate_next_population()

        else: # We create different Targets around the Agent, depending on its Position (ignoring the input angle):
            circle = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2,  7*np.pi/4]
            pos_target = []
            scalar = [.5, 2, 1, 1, 1.5, .5, 2, 1.5]

            for j,cle in enumerate(circle):
                tpos = np.array(position_agent) + np.array([np.cos(cle), np.sin(cle)]) * distance * scalar[j]
                pos_target.append(tpos)

            return pos_target


    def _simulate_next_population(self, position_agent, pos_target):
        '''
        Run simulation => fitness
        We save the distance to (each) target. The fitness will be the (average) distance
        If we have more than one target:
            - each agent will run through all trials (each trial the target is on a different position).
            - we take average Fitness over all  ('complex trials')

        :param position_agent:
        :param pos_target:
        :return: Updates sorted pop_list
        '''

        assert self.pop_list[-1,1] == 0, "This population run already its simulation"

        for i,string in enumerate(self.pop_list):  # Run simulation with each agent

            genome_string = string[2:]

            Fitness = []

            for tpos in pos_target:

                self.agent = CatchBot(position_agent=position_agent, position_target=[tpos[0],tpos[1]])  # reset self.agent and set new target position

                self.implement_genome(genome_string)      # implement the current genome in agent

                self.agent.movement(self.simlength)

                Fitness.append(self.fitness())

            self.pop_list[i,1] = np.sum(Fitness)/len(Fitness)       # agent's average fitness will be stored

        self.pop_list = copy.copy(mat_sort(self.pop_list, index=1))


    def run_evolution(self, Generations, mutation_var = .25, complex_trials=True, fit_prop_sel = False, position_agent=[50,50], angle_to_target= np.pi/2,  distance_to_target = 30):

        # Ask whether results should be saved in external file
        save = save_request()

        # Run evolution:
        Fitness_progress = np.zeros((Generations,3))

        pos_target = self._set_target(position_agent=position_agent,
                                      angle_to_target=angle_to_target,
                                      distance=distance_to_target,
                                      complex=complex_trials)

        for i in range(Generations):

            self._reproduction(mutation_var, fps=fit_prop_sel)

            self._simulate_next_population(position_agent = position_agent,
                                           pos_target     = pos_target)



            Fitness_progress[i,1:] = np.round(self.pick_best()[:,1],2)

            self.Generation += 1

            Fitness_progress[i,0]  = self.Generation

            print(Fitness_progress[i,1:], "Generation", self.Generation)


        # Save in external file:
        if save:

            self.filename = "sim{}.mut{}.Gen{}-{}_CT={}.fps={}".format(self.simlength, mutation_var, self.Generation - Generations + 1,
                                                                       self.Generation, complex_trials,fit_prop_sel)

            pickle.dump(self.pop_list,                open('Poplist.{}'.format(self.filename), 'wb'))
            pickle.dump(np.round(Fitness_progress,2), open('Fitness_progress.{}'.format(self.filename), 'wb'))

            print('Evolution terminated. pop_list saved \n'
                  '(Filename: "Poplist.{}")'.format(self.filename))
        else:
            print('Evolution terminated. \n'
                  '(Caution: pop_list is not saved in external file)')


        return Fitness_progress


    def reimplement_population(self, Filename=None, Plot = False):

        if Filename is None:
            Filename = self.filename
            print("Reimplements its own pop_list file")


        # Reimplement: pop_list, simlength, Generation
        self.pop_list        = pickle.load(open('Poplist.{}'.format(Filename),          'rb'))

        self.simlength       = int(Filename[Filename.find('m')+1 : Filename.find('.')])  # depends on filename

        Fitness_progress     = pickle.load(open('Fitness_progress.{}'.format(Filename), 'rb'))
        self.Generation      = int(Fitness_progress[-1,0])


        if Plot:

            # here we plot the fitness progress of all generation
            plt.figure()
            plt.plot(Fitness_progress[:,1])
            plt.plot(Fitness_progress[:,2])

            # Here we plot the trajectory of the best agent:
            self.plot_pop_list()
            print("Plot the best agent")

            global n   # this is needed for self.close()
            n = 2


    def plot_pop_list(self, n_agents=1, position_agent=[50,50]):

        global n
        n = n_agents
        pos_target = self._set_target(position_agent=position_agent, complex=True)

        for i in range(n_agents):

            plt.figure()

            for tpos in pos_target:
                self.agent = CatchBot(position_agent = position_agent)
                self.agent.position_target = tpos
                self.implement_genome(self.pop_list[i,2:])
                self.run_and_plot()
                # TODO: plot the same colour as the assocciaed trajectory
                plt.plot(tpos[0], tpos[1], 'ws')

            plt.plot(position_agent[0], position_agent[1], 'bo')

        print(np.round(self.pop_list[0:n_agents, 0:3],2))
        if n_agents > 1:
            print("Close all Windows with close()")


    def close(self):
        for j in range(n):  # n is from the global variable of plot_pop_list()/reimplement_population()
            plt.close()



# t3 = Evolution(simlength=50)
# t3.run_evolution(Generations=10)
# t3.plot_pop_list(2)
# t3.close()
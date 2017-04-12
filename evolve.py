import pickle
import numpy as np
import random
import math
from copy import deepcopy
import CTRNN
import simulate


class Evolution:
    def __init__(self, pop_size, evolution_params, network_params, evaluation_params, agent_params):
        """
        Class that executes genetic search.
        :param pop_size: size of the new population
        :param evolution_params: evolution parameters for the simulation
        :param network_params: neural network parameters for the simulation
        :param evaluation_params: parameters used for running trials during the evaluation
        """
        self.pop_size = pop_size
        self.max_gens = evolution_params['max_gens']
        self.mutation_var = evolution_params['mutation_variance']
        self.prob_crossover = evolution_params['prob_crossover']
        self.elitist_fraction = evolution_params['elitist_fraction']
        self.fps_fraction = evolution_params['fps_fraction']
        self.check_int = evolution_params['check_int']
        self.step_size = network_params['step_size']
        self.network_params = network_params
        self.evaluation_params = evaluation_params
        self.agent_params = agent_params

    def run(self):
        """
        Execute a full search run until some condition is reached.
        :return: the last population in the search
        """
        gen = 0
        # create initial population
        population = self.create_population(self.pop_size)

        # save the initial population
        popfile = open('./Agents/gen0', 'wb')
        pickle.dump(population, popfile)
        popfile.close()

        # collect average and best fitness
        avg_fitness = []
        best_fitness = []

        while gen < self.max_gens:
            # print(gen)
            # evaluate all agents on the task
            for agent in population:
                simulation_run = simulate.Simulation(self.step_size, self.evaluation_params)
                trial_data = simulation_run.run_trials(agent, simulation_run.trials)  # returns a list of fitness in all trials
                agent.fitness = np.mean(trial_data['fitness'])

            # log fitness results
            population_avg_fitness = np.mean([agent.fitness for agent in population])
            avg_fitness.append(round(population_avg_fitness, 3))
            best_fitness.append(round(max([agent.fitness for agent in population]), 3))

            # reproduce population
            population = self.reproduce(population)

            # save the intermediate population
            if gen % self.check_int == 0:
                popfile = open('./Agents/gen{}'.format(gen), 'wb')
                pickle.dump(population, popfile)
                popfile.close()
                print("Saved generation {}".format(gen))

            gen += 1

        fits = [avg_fitness, best_fitness]
        fit_file = open('./Agents/fitnesses', 'wb')
        pickle.dump(fits, fit_file)
        fit_file.close()

    def create_population(self, size):
        """
        Create random population: used for creating a random initial population and random portion of the new population
        in each generation.
        :return: population of agents
        """
        population = []
        for i in range(size):
            # create the agent's CTRNN brain
            agent_brain = CTRNN.CTRNN(self.network_params['num_neurons'],
                                      self.network_params['step_size'],
                                      self.network_params['tau_range'],
                                      self.network_params['g_range'],
                                      self.network_params['theta_range'],
                                      self.network_params['w_range'])
            # create new agent
            agent = simulate.Agent(agent_brain, self.agent_params)
            population.append(agent)
        return population

    def reproduce(self, population):
        """
        Reproduce a single generation in the following way:
        1) Copy the proportion equal to elitist_fraction of the current population to the new population (these are best_agents)
        2) Select the rest of the population for crossover using fitness proportionate selection (FPS), excluding the best_agents
        3) Shuffle the selected population in preparation for cross-over
        4) Create crossover_fraction children of selected population with probability of crossover equal to prob_crossover.
        Crossover takes place at genome module boundaries (single neurons).
        5) Apply mutation to the children with mutation equal to mutation_var
        6) Fill the rest of the population with randomly created agents

        :param population: the population to be reproduced
        :return: new_population
        """

        new_population = [None] * self.pop_size

        # sort agents by fitness from best to worst
        population.sort(key=lambda agent: agent.fitness, reverse=True)
        # print("Best fitness in this generation is {}".format(population[0].fitness))

        # calculate all fractions
        n_best = math.floor(self.pop_size * self.elitist_fraction + 0.5)
        n_crossed = int(math.floor(self.pop_size * self.fps_fraction + 0.5)) & (-2)  # floor to the nearest even number
        n_fillup = self.pop_size - (n_best + n_crossed)

        # 1) Elitist selection

        best_agents = deepcopy(population[:n_best])
        new_population[:n_best] = best_agents
        newpop_counter = n_best  # track where we are in the new population

        # 2) Select mating population

        mating_pool = self.select_mating_pool(population)

        # 3) Shuffle
        random.shuffle(mating_pool)

        # 4, 5) Create children with crossover or apply mutation
        mating_counter = 0
        mating_finish = newpop_counter + n_crossed

        while newpop_counter < mating_finish:
            r = np.random.random()
            parent1 = mating_pool[mating_counter]
            parent2 = mating_pool[mating_counter + 1]

            if r < self.prob_crossover and parent1 != parent2:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                # if the two parents are the same, mutate them to get children
                child1 = self.mutate(parent1, self.mutation_var)
                child2 = self.mutate(parent2, self.mutation_var)
            new_population[newpop_counter], new_population[newpop_counter+1] = child1, child2
            newpop_counter += 2
            mating_counter += 2

        # 6) Fill up with random new agents
        new_population[newpop_counter:] = self.create_population(n_fillup)

        return new_population

    @staticmethod
    def select_mating_pool(population):
        """
        Select a mating pool population based on fitness proportionate selection method.
        The probability of an individual being selected for reproduction is agent.fitness/total_population_fitness
        :param population: the population from which to select the parents 
        :return: selected parents for reproduction
        """
        fitnesses = [agent.fitness for agent in population]
        total_fitness = float(sum(fitnesses))
        rel_fitness = [f/total_fitness for f in fitnesses]

        # Generate probability intervals for each individual
        probs = [sum(rel_fitness[:i + 1]) for i in range(len(rel_fitness))]
        # Draw new population
        new_population = []
        for _ in range(len(population)):
            r = np.random.random()
            for (i, agent) in enumerate(population):
                if r <= probs[i]:
                    new_population.append(agent)
                    break
        return new_population

    @staticmethod
    def crossover(parent1, parent2):
        """
        Given two agents, create two new agents by exchanging their genetic material.
        :param parent1: first parent agent
        :param parent2: second parent agent
        :return: two new agents
        """
        crossover_point = np.random.choice(parent1.crossover_points)
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        child1.genotype = np.hstack((parent1.genotype[:crossover_point], parent2.genotype[crossover_point:]))
        child2.genotype = np.hstack((parent2.genotype[:crossover_point], parent1.genotype[crossover_point:]))
        child1.make_params_from_genotype(child1.genotype)
        child2.make_params_from_genotype(child2.genotype)
        return child1, child2

    def mutate(self, agent, mutation_var):
        magnitude = np.random.normal(0, np.sqrt(mutation_var))
        unit_vector = np.array(self.make_rand_vector(len(agent.genotype)))
        mutant = deepcopy(agent)
        mutant.genotype = np.clip(agent.genotype + magnitude * unit_vector,
                                  self.agent_params['gene_range'][0], self.agent_params['gene_range'][1])
        mutant.make_params_from_genotype(mutant.genotype)
        return mutant

    @staticmethod
    def make_rand_vector(dims):
        """
        Generate a random unit vector.  This works by first generating a vector each of whose elements 
        is a random Gaussian and then normalizing the resulting vector.
        """
        vec = np.random.normal(0, 1, dims)
        mag = sum(x ** 2 for x in vec) ** .5
        return [x / mag for x in vec]


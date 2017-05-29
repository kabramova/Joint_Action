import pickle
import numpy as np
import random
import math
from copy import deepcopy
import CTRNN
import simulate
np.seterr(over='ignore')


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
        self.evolution_params = evolution_params
        self.step_size = network_params['step_size']
        self.network_params = network_params
        self.evaluation_params = evaluation_params
        self.agent_params = agent_params

    def run(self, gen_to_load):
        """
        Execute a full search run until some condition is reached.
        :param gen_to_load: which generation to load if starting from an existing population
        :return: the last population in the search
        """
        gen = 0
        # create initial population or load existing one
        if gen_to_load is None:
            population = self.create_population(self.pop_size)
        else:
            population = self.load_population(gen)

        # collect average and best fitness
        avg_fitness = [0]
        best_fitness = [0]
        best_counter = 0

        while gen < self.evolution_params['max_gens'] + 1:
            #  print(gen)
            # evaluate all agents on the task
            for agent in population:
                # initialize a type of simulation
                simulation_run = simulate.Simulation(self.step_size, self.evaluation_params)
                # simulation_run = simulate.SimpleSimulation(self.step_size, self.evaluation_params)

                # run the trials and return fitness in all trials
                trial_data = simulation_run.run_trials(agent, simulation_run.trials)

                # calculate overall fitness
                # agent.fitness = np.mean(trial_data['fitness'])
                agent.fitness = self.harmonic_mean(trial_data['fitness'])
                # agent.fitness = min(trial_data['fitness'])

            # log fitness results: average population fitness
            population_avg_fitness = np.mean([agent.fitness for agent in population])
            avg_fitness.append(round(population_avg_fitness, 3))

            # sort agents by fitness from best to worst
            population.sort(key=lambda ag: ag.fitness, reverse=True)
            # log fitness results: best agent fitness
            bf = round(population[0].fitness, 3)
            best_fitness.append(bf)

            # stop the search if fitness hasn't increased in a set number of generations
            if bf == best_fitness[-2]:
                best_counter += 1

                if best_counter > self.evolution_params['evolution_break']:
                    # save the last population
                    self.save_population(population, gen)
                    print("Stopped the search at generation {}".format(gen))

                    # save the average and best fitness lists
                    self.log_fitness(avg_fitness, best_fitness)
                    break
            else:
                best_counter = 0

            # save the intermediate or last population and fitness
            if gen % self.evolution_params['check_int'] == 0 or gen == self.evolution_params['max_gens']:
                self.save_population(population, gen)
                print("Saved generation {}".format(gen))
                self.log_fitness(avg_fitness, best_fitness)

            # reproduce population
            population = self.reproduce(population)
            gen += 1

    def create_population(self, size):
        """
        Create random population: used for creating a random initial population and random portion 
        of the new population in each generation.
        :param size: the size of the population to create
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

            if self.evaluation_params['velocity_control'] == "direct":
                agent = simulate.DirectVelocityAgent(agent_brain, self.agent_params, self.evaluation_params['screen_width'])
            else:
                # create new agent of a certain type
                # agent = simulate.Agent(agent_brain, self.agent_params)
                # agent = simulate.EmbodiedAgentV1(agent_brain, self.agent_params, self.evaluation_params['screen_width'])
                agent = simulate.EmbodiedAgentV2(agent_brain, self.agent_params, self.evaluation_params['screen_width'])
                # agent = simulate.ButtonOnOffAgent(agent_brain, self.agent_params, self.evaluation_params['screen_width'])
            population.append(agent)
        return population

    @staticmethod
    def load_population(gen):
        pop_file = open('./Agents/gen{}'.format(gen), 'rb')
        population = pickle.load(pop_file)
        pop_file.close()
        population.sort(key=lambda agent: agent.fitness, reverse=True)
        return population

    @staticmethod
    def save_population(population, gen):
        pop_file = open('./Agents/gen{}'.format(gen), 'wb')
        pickle.dump(population, pop_file)
        pop_file.close()

    @staticmethod
    def log_fitness(avg, best):
        fits = [avg, best]
        fit_file = open('./Agents/fitnesses', 'wb')
        pickle.dump(fits, fit_file)
        fit_file.close()

    def reproduce(self, population):
        """
        Reproduce a single generation in the following way:
        1) Copy the proportion equal to elitist_fraction of the current population to the new population (these are best_agents)
        2) Select part of the population for crossover using some selection method (set in config)
        3) Shuffle the selected population in preparation for cross-over
        4) Create crossover_fraction children of selected population with probability of crossover equal to prob_crossover.
        Crossover takes place at genome module boundaries (single neurons).
        5) Apply mutation to the children with mutation equal to mutation_var
        6) Fill the rest of the population with randomly created agents

        :param population: the population to be reproduced
        :return: new_population
        """

        new_population = [None] * self.pop_size

        # calculate all fractions
        n_best = math.floor(self.pop_size * self.evolution_params['elitist_fraction'] + 0.5)
        n_crossed = int(math.floor(self.pop_size * self.evolution_params['fps_fraction'] + 0.5)) & (-2)  # floor to the nearest even number
        n_fillup = self.pop_size - (n_best + n_crossed)

        # 1) Elitist selection

        best_agents = deepcopy(population[:n_best])
        new_population[:n_best] = best_agents
        newpop_counter = n_best  # track where we are in the new population

        # 2) Select mating population from the remaining population

        updated_fitness = self.update_fitness(population, self.evolution_params['fitness_update'], 1.1)
        mating_pool = self.select_mating_pool(population, updated_fitness, n_crossed, self.evolution_params['selection'])

        # 3) Shuffle
        random.shuffle(mating_pool)

        # 4, 5) Create children with crossover or apply mutation
        mating_counter = 0
        mating_finish = newpop_counter + n_crossed

        while newpop_counter < mating_finish:
            if (mating_finish - newpop_counter) == 1:
                new_population[newpop_counter] = self.mutate(mating_pool[mating_counter], self.evolution_params['mutation_variance'])
                newpop_counter += 1
                mating_counter += 1

            else:
                r = np.random.random()
                parent1 = mating_pool[mating_counter]
                parent2 = mating_pool[mating_counter + 1]

                if r < self.evolution_params['prob_crossover'] and parent1 != parent2:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    # if the two parents are the same, mutate them to get children
                    child1 = self.mutate(parent1, self.evolution_params['mutation_variance'])
                    child2 = self.mutate(parent2, self.evolution_params['mutation_variance'])
                new_population[newpop_counter], new_population[newpop_counter+1] = child1, child2
                newpop_counter += 2
                mating_counter += 2

        # 6) Fill up with random new agents
        new_population[newpop_counter:] = self.create_population(n_fillup)

        return new_population

    @staticmethod
    def update_fitness(population, method, max_exp_offspring=None):
        """
        Update agent fitness to relative values, retain sorting from best to worst.
        :param population: the population whose fitness needs updating
        :param method: fitness proportionate or rank-based
        :param max_exp_offspring: 
        :return: 
        """
        rel_fitness = []
        if method == 'fps':
            fitnesses = [agent.fitness for agent in population]
            total_fitness = float(sum(fitnesses))
            rel_fitness = [f/total_fitness for f in fitnesses]

        elif method == 'rank':
            # Baker's linear ranking method: f(pos) = 2-SP+2*(SP-1)*(pos-1)/(n-1)
            # the highest ranked individual receives max_exp_offspring (typically 1.1), the lowest receives 2 - max_exp_offspring
            # normalized to sum to 1
            ranks = list(range(1, len(population)+1))
            rel_fitness = [(max_exp_offspring + (2 - 2 * max_exp_offspring) * (ranks[i]-1) / (len(population)-1)) / len(population)
                           for i in range(len(population))]

        elif method == 'sigma':
            # for every individual 1 + (I(f) - P(avg_f))/2*P(std) is calculated
            # if value is below zero, a small positive constant is given so the individual has some probability
            # of being chosen. The numbers are then normalized
            fitnesses = [agent.fitness for agent in population]
            avg = np.mean(fitnesses)
            std = max(0.0001, np.std(fitnesses))
            exp_values = list((1 + ((f - avg) / (2 * std))) for f in fitnesses)

            for i, v in enumerate(exp_values):
                if v <= 0:
                    exp_values[i] = 1 / len(population)
            s = sum(exp_values)
            rel_fitness = list(e / s for e in exp_values)

        return rel_fitness

    @staticmethod
    def select_mating_pool(population, updated_fitness, n_parents, method):
        """
        Select a mating pool population.
        :param population: the population from which to select the parents
        :param updated_fitness: the relative updated fitness
        :param n_parents: how many parents to select
        :param method: which method to use for selection (roulette wheel or stochastic universal sampling)
        :return: selected parents for reproduction
        """
        new_population = []

        if method == "rws":
            # roulette wheel selection
            probs = [sum(updated_fitness[:i + 1]) for i in range(len(updated_fitness))]
            # Draw new population
            new_population = []
            for _ in range(n_parents):
                r = np.random.random()
                for (i, agent) in enumerate(population):
                    if r <= probs[i]:
                        new_population.append(agent)
                        break

        elif method == "sus":
            # stochastic universal sampling selection
            probs = [sum(updated_fitness[:i + 1]) for i in range(len(updated_fitness))]
            p_dist = 1/n_parents  # distance between the pointers
            start = np.random.uniform(0, p_dist)
            pointers = [start + i*p_dist for i in range(n_parents)]

            for p in pointers:
                for (i, agent) in enumerate(population):
                    if p <= probs[i]:
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

    @staticmethod
    def harmonic_mean(fit_list):
        if 0 in fit_list:
            return 0
        else:
            return len(fit_list) / np.sum(1.0 / np.array(fit_list))

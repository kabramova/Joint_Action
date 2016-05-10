from JA_Simulator import *

class JA_Evolution(JA_Simulation):

    def __init__(self):

        super(self.__class__, self).__init__(simlength=2789)


    def run_trials(self):

        fitness_per_trials = []

        for trial_speed in ["slow", "fast"]:
            for auditory_condition in [False, True]:
                for init_target_direction in [-1, 1]:  # left(-1) or right(1)

                    self.setup(trial_speed=trial_speed, auditory_condition=auditory_condition)
                    self.target.velocity *= init_target_direction
                    fitness = self.run()
                    fitness_per_trials.append(fitness)

        fitness = np.mean(fitness_per_trials)
        print("Average fitness over all trials:", fitness)

        return fitness


    # TODO: Trajectories, Genome

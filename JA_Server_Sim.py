from JA_Evolution import *

ja = JA_Evolution(auditory_condition=True)
ja.run_evolution(generations=3)

ja2 = JA_Evolution(auditory_condition=False)
ja2.run_evolution(generations=100)



## Test
# ja = JA_Evolution(pop_size=5, auditory_condition=False)
# print("Sound_Cond:", ja.condition,", popsize:", ja.pop_size,", Gen:", ja.generation)
# ja.reimplement_population(filename="Gen1-3.popsize111.mut0.02.sound_cond=True.JA.single", Plot=False)
# print("Sound_Cond:", ja.condition,", popsize:", ja.pop_size,", Gen:", ja.generation)
# ja.print_best()

# sim = JA_Simulation(False)
# sim.setup(trial_speed="slow")
# print("Simlength:", sim.simlength)
# sim_table = sim.run_and_plot() # sim_table contains: fitness[0], trajectories[1], keypress[2] and sounds[3](if applicable)


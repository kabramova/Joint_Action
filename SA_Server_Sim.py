from SA_Evolution import *

audicon = audio_condition_request()
number_of_generations = generation_request()
print("Run Evolution for {} Generations in Sound Condition={}".format(number_of_generations, audicon))

sa = SA_Evolution(auditory_condition=audicon)
sa.run_evolution(generations=number_of_generations)



## Test
# sa = SA_Evolution(pop_size=5, auditory_condition=False)
# print("Sound_Cond:", sa.condition,", popsize:", sa.pop_size,", Gen:", sa.generation)
# sa.reimplement_population(filename="Gen1-2.popsize111.mut0.02.sound_cond=False.JA.single(Fitness9.99)", Plot=True)
# print("Sound_Cond:", sa.condition,", popsize:", sa.pop_size,", Gen:", sa.generation)
# sa.print_best(5)

# sim = SA_Simulation(False)
# sim.setup(trial_speed="slow")
# print("Simlength:", sim.simlength)
# sim_table = sim.run_and_plot() # sim_table contains: fitness[0], trajectories[1], keypress[2] and sounds[3](if applicable)


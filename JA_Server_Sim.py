from JA_Evolution import *

audicon = audio_condition_request()
number_of_generations = generation_request()
print("Run Evolution for {} Generations in Sound Condition={}".format(number_of_generations, audicon))

ja = JA_Evolution(auditory_condition=audicon)
ja.run_evolution(generations=number_of_generations)



## Test
# ja = JA_Evolution(pop_size=5, auditory_condition=False)
# print("Sound_Cond:", ja.condition,", popsize:", ja.pop_size,", Gen:", ja.generation)
# ja.reimplement_population(filename="Gen1-2.popsize111.mut0.02.sound_cond=False.JA.single(Fitness9.99)", Plot=True)
# print("Sound_Cond:", ja.condition,", popsize:", ja.pop_size,", Gen:", ja.generation)
# ja.print_best(5)

# sim = JA_Simulation(False)
# sim.setup(trial_speed="slow")
# print("Simlength:", sim.simlength)
# sim_table = sim.run_and_plot() # sim_table contains: fitness[0], trajectories[1], keypress[2] and sounds[3](if applicable)


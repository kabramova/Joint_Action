from JA_Evolution import *

audit = input("Sound condition is '(T)rue' / '(F)alse':")

if audit ==1 or audit.lower() =="t" or audit.lower() =="true":
    audit = True
elif audit ==0 or audit.lower() =="f" or audit.lower() =="false":
    audit = False
else:
    raise ValueError("Must be True or False")

gens = input("How many Generations to run (int):")
if int(gens): gens = int(gens)

print("Run Evolution for {} Generations in Sound Condition={}".format(gens, audit))

ja = JA_Evolution(auditory_condition=audit)
ja.run_evolution(generations=gens)



## Test
ja = JA_Evolution(pop_size=5, auditory_condition=False)
print("Sound_Cond:", ja.condition,", popsize:", ja.pop_size,", Gen:", ja.generation)
ja.reimplement_population(filename="Gen1-2.popsize111.mut0.02.sound_cond=False.JA.single(Fitness9.99)", Plot=True)
print("Sound_Cond:", ja.condition,", popsize:", ja.pop_size,", Gen:", ja.generation)
ja.print_best(5)

# sim = JA_Simulation(False)
# sim.setup(trial_speed="slow")
# print("Simlength:", sim.simlength)
# sim_table = sim.run_and_plot() # sim_table contains: fitness[0], trajectories[1], keypress[2] and sounds[3](if applicable)


from SA_Evolution import *

# For the CPU split:
# Type in Terminal (-P*, * must be equal to n_cpu):
#  cat args_splitter | xargs -L1 -P6 python3 SA_Server_Sim.py
# Note: args_splitter must contain numbers from 1 to n_cpu

n_cpu = 6

if len(sys.argv) > 1 and sys.argv[1].isdigit():
        split = int(sys.argv[1]) if int(sys.argv[1]) <= n_cpu else False
else:
    split = False


if not split:  # is False
    audicon = audio_condition_request()
    number_of_generations = generation_request()
    filename = filename_request("single")
    scalar = simlength_scalar_request()

else:  # if splitter is used, these values must be pre-given, here in python file
    # Manually adjust the following parameters:
    audicon = False
    number_of_generations = 1000
    scalar = 1
    filename = "Gen20001-21000.popsize111.mut0.02.sound_cond=False.JA.single(Fitness6.98)"  # or None
    print("Splitter {} started!".format(split))


sa = SA_Evolution(auditory_condition=audicon, pop_size=110, simlength_scalar=scalar)


if isinstance(filename, str):
    sa.reimplement_population(filename=filename, plot=False)
    if not split or split == n_cpu:
        if audicon != sa.condition:
            print("...")
            print("Note: Initial Sound Condition differs from the one in implemented file!")
            print("...")


# RUN:
if not split or split == n_cpu:
    print("Run Evolution for {} Generations in Single Condition and Sound Condition={}".format(number_of_generations,
                                                                                               sa.condition))
sa.run_evolution(generations=number_of_generations, splitter=split, n_cpu=n_cpu)


# # Reimplement and Plot
# sa2 = SA_Evolution(auditory_condition=False)
# print("Sound_Cond:", sa2.condition,", Popsize:", sa2.pop_size,", Gen:", sa2.generation)
# filename = filename_request("single")
# output = sa2.reimplement_population(filename=filename, plot=True)
# print("Sound_Cond:", sa2.condition,", Popsize:", sa2.pop_size,", Gen:", sa2.generation)
# sa2.print_best(5)
# # print(np.round(sa2.pop_list[0:50,0:2],2))

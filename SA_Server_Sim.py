from SA_Evolution import *

## For the CPU split:
# Type in Terminal:
#  cat args_splitter | xargs -L1 -P6 python3 SA_Server_Sim.py

n_cpu = 6

if len(sys.argv) > 1 and sys.argv[1].isdigit():
        split = int(sys.argv[1]) if int(sys.argv[1]) <= n_cpu else False
else:
    split = False


if split == False:
    audicon = audio_condition_request()
    number_of_generations = generation_request()
    filename = filename_request("single")

else: # if splitter is used, these values must be pre-given, here in python file
    audicon = False
    number_of_generations = 1000
    filename = "Gen1501-2000.popsize111.mut0.02.sound_cond=False.JA.single(Fitness6.73)"
    print("Splitter {} started!".format(split))
    if split == 6:
        print("Run Evolution for {} Generations in Sound Condition={}".format(number_of_generations, audicon))


sa = SA_Evolution(auditory_condition=audicon)


if split == False or split == n_cpu:
    if isinstance(filename, str):
        sa.reimplement_population(filename=filename, Plot=False)
        print("...")
        print("File is successfully implemented")

    if audicon != sa.condition:
        print("...")
        print("Note: Initial Sound Condition differs from the one in implemented file!")
        print("...")

        print("Run Evolution for {} Generations in Sound Condition={}".format(number_of_generations, sa.condition))


# RUN:
sa.run_evolution(generations=number_of_generations, splitter=split)


# # Reimplement and Plot
# sa2 = SA_Evolution(auditory_condition=False)
# print("Sound_Cond:", sa2.condition,", Popsize:", sa2.pop_size,", Gen:", sa2.generation)
# filename = filename_request("single")
# output = sa2.reimplement_population(filename=filename, Plot=True)
# print("Sound_Cond:", sa2.condition,", Popsize:", sa2.pop_size,", Gen:", sa2.generation)
# sa2.print_best(5)
# # print(np.round(sa2.pop_list[0:50,0:2],2))

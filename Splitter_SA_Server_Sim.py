from Splitter_SA_Evolution import *

## For the CPU split:
#  cat args_splitter | xargs -L1 -P6 python3 Splitter_SA_Server_Sim.py (in Terminal)

if len(sys.argv) > 1 and sys.argv[1].isdigit():
        split = int(sys.argv[1]) if int(sys.argv[1]) <= 6 else False
else:
    split = False


if split == False:
    audicon = audio_condition_request()
    number_of_generations = generation_request()
    filename = filename_request("single")
else:
    audicon = False             # if splitter is used, these values must be pre-given, here in python file
    number_of_generations = 10
    filename = "Gen1501-2000.popsize111.mut0.02.sound_cond=False.JA.single(Fitness6.73)"
    print("Splitter {} started!".format(split))

sa = SA_Evolution(auditory_condition=audicon)

if isinstance(filename, str):
    sa.reimplement_population(filename=filename, Plot=False)
    print("...")
    print("File is successfully implemented")

if audicon != sa.condition:
    print("...")
    print("Note: Initial Sound Condition differs from the one in implemented file!")
    print("...")

print("Run Evolution for {} Generations in Sound Condition={}".format(number_of_generations, sa.condition))

sa.run_evolution(generations=number_of_generations, splitter=split)

#TODO: create seperate split-folders for temporary files:

# # Reimplement and Plot
# sa2 = SA_Evolution(auditory_condition=False)
# print("Sound_Cond:", sa2.condition,", Popsize:", sa2.pop_size,", Gen:", sa2.generation)
# filename = filename_request("single")
# output = sa2.reimplement_population(filename=filename, Plot=True)
# print("Sound_Cond:", sa2.condition,", Popsize:", sa2.pop_size,", Gen:", sa2.generation)
# sa2.print_best(5)
# # print(np.round(sa2.pop_list[0:50,0:2],2))

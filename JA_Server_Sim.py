from JA_Evolution import *

## For the CPU split:
# Type in Terminal:
#  cat args_splitter | xargs -L1 -P6 python3 JA_Server_Sim.py

#TODO: Solve: ..._request()'s issue with xargs

#TODO: funzt nog niet

n_cpu = 6

if len(sys.argv) > 1 and sys.argv[1].isdigit():
    split = int(sys.argv[1]) if int(sys.argv[1]) <= n_cpu else False
else:
    split = False

if split == False:
    audicon = audio_condition_request()
    number_of_generations = generation_request()
    filename = filename_request("joint")

else: # if splitter is used, these values must be pre-given, here in python file
    audicon = False
    number_of_generations = 1000
    filename = "Gen501-1000.popsize55.mut0.02.sound_cond=False.JA.joint(Fitness6.1)"
    print("Splitter {} started!".format(split))



ja = JA_Evolution(auditory_condition=audicon, pop_size=55)

if isinstance(filename, str):
    ja.reimplement_population(filename=filename, Plot=False)


if split == False or split == n_cpu:
    print("...")
    print("File is successfully implemented")

    if audicon != ja.condition:
        print("...")
        print("Note: Initial Sound Condition differs from the one in implemented file!")
        print("...")

    print("Run Evolution for {} Generations in Sound Condition={}".format(number_of_generations, audicon))

# RUN:
ja.run_evolution(generations=number_of_generations, splitter=split)

# # Reimplement and Plot
# ja2 = JA_Evolution(auditory_condition=True)
# print("Sound_Cond:", ja2.condition,", Popsize:", ja2.pop_size,", Gen:", ja2.generation)
# filename = filename_request("joint")
# output = ja2.reimplement_population(filename=filename, Plot=True)
# print("Sound_Cond:", ja2.condition,", Popsize:", ja2.pop_size,", Gen:", ja2.generation)
# ja2.print_best(5)
# # print(np.round(ja2.pop_list_L[0:50,0:2],2))
# # print(np.round(ja2.pop_list_R[0:50,0:2],2))


#TODO: SAVE files in bin Evolution.py, SA_Evolution, JA_Evolution seperately.
#TODO: Analysis:
# - Behaviour compare with paper
# - analysis of strategy.
# - analysis William Beer 2015 (p.8)
# - Attractor space
# - Statistic for meaningful difference between sound off/on (run evolution n-times, for x-Generations)

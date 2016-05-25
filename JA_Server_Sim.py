# TODO: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

from JA_Evolution import *

audicon = audio_condition_request()
number_of_generations = generation_request()

ja = JA_Evolution(auditory_condition=audicon, pop_size=10)

filename = filename_request("joint")

if isinstance(filename, str):
    ja.reimplement_population(filename=filename, Plot=False)
    print("...")
    print("File is successfully implemented")

if audicon != ja.condition:
    print("...")
    print("Note: Initial Sound Condition differs from the one in implemented file!")
    print("...")

print("Run Evolution for {} Generations in Sound Condition={}".format(number_of_generations, ja.condition))
ja.run_evolution(generations=number_of_generations)


# Reimplement and Plot
ja2 = JA_Evolution(auditory_condition=True)
print("Sound_Cond:", ja2.condition,", Popsize:", ja2.pop_size,", Gen:", ja2.generation)
filename = filename_request("joint")
output = ja2.reimplement_population(filename=filename, Plot=True)
print("Sound_Cond:", ja2.condition,", Popsize:", ja2.pop_size,", Gen:", ja2.generation)
ja2.print_best(5)
# print(np.round(ja2.pop_list_L[0:50,0:2],2))
# print(np.round(ja2.pop_list_R[0:50,0:2],2))





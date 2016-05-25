from SA_Evolution import *

audicon = audio_condition_request()
number_of_generations = generation_request()

sa = SA_Evolution(auditory_condition=audicon)

filename = filename_request("single")

if isinstance(filename, str):
    sa.reimplement_population(filename=filename, Plot=False)
    print("...")
    print("File is successfully implemented")

if audicon != sa.condition:
    print("...")
    print("Note: Initial Sound Condition differs from the one in implemented file!")
    print("...")

print("Run Evolution for {} Generations in Sound Condition={}".format(number_of_generations, sa.condition))
sa.run_evolution(generations=number_of_generations)


# # Reimplement and Plot
# sa2 = SA_Evolution(auditory_condition=False)
# print("Sound_Cond:", sa2.condition,", Popsize:", sa2.pop_size,", Gen:", sa2.generation)
# filename = filename_request("single")
# output = sa2.reimplement_population(filename=filename, Plot=True)
# print("Sound_Cond:", sa2.condition,", Popsize:", sa2.pop_size,", Gen:", sa2.generation)
# sa2.print_best(5)
# # print(np.round(sa2.pop_list[0:50,0:2],2))

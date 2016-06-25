from JA_Evolution import *

# For the CPU split:
# Type in Terminal (-P*, * must be equal to n_cpu):
#  cat args_splitter | xargs -L1 -P6 python3 JA_Server_Sim.py
# note: args_splitter must contain numbers from 1 to n_cpu

n_cpu = 6

if len(sys.argv) > 1 and sys.argv[1].isdigit():
    split = int(sys.argv[1]) if int(sys.argv[1]) <= n_cpu else False
else:
    split = False

if not split:  # is False
    audicon = audio_condition_request()
    number_of_generations = generation_request()
    filename = filename_request("joint")
    scalar = simlength_scalar_request()

else:  # if splitter is used, these values must be pre-given, here in python file
    # Manually adjust the following parameters:
    audicon = True
    number_of_generations = 2000
    scalar = 1
    filename = "Gen14501-19000.popsize55.mut0.02.sound_cond=True.JA.joint(Fitness6.02)"  # or None
    print("Splitter {} started!".format(split))


ja = JA_Evolution(auditory_condition=audicon, pop_size=55, simlength_scalar=scalar)


if isinstance(filename, str):
    ja.reimplement_population(filename=filename, Plot=False)
    if not split or split == n_cpu:
        if audicon != ja.condition:
            print("...")
            print("Note: Initial Sound Condition differs from the one in implemented file!")
            print("...")


# RUN:
if not split or split == n_cpu:
    print("Run Evolution for {} Generations in Joint Condition and Sound Condition={}".format(number_of_generations,
                                                                                              ja.condition))
ja.run_evolution(generations=number_of_generations, splitter=split, n_cpu=n_cpu)


# Reimplement and plot:
# ja2 = JA_Evolution(auditory_condition=True)
# print("Sound_Cond:", ja2.condition,", Popsize:", ja2.pop_size,", Gen:", ja2.generation)
# filename = filename_request("joint")
# output = ja2.reimplement_population(filename=filename, Plot=True)
# print("Sound_Cond:", ja2.condition,", Popsize:", ja2.pop_size,", Gen:", ja2.generation)
# ja2.print_best(5)
# print(np.round(ja2.pop_list_L[0:50,0:2],2))
# print(np.round(ja2.pop_list_R[0:50,0:2],2))

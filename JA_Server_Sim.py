from JA_Evolution import *

"""
__author__  = Simon Hofmann"
__credits__ = ["Simon Hofmann", "Katja Abramova", "Willem Zuidema"]
__version__ = "1.0.1"
__date__ "2016"
__maintainer__ = "Simon Hofmann"
__email__ = "simon.hofmann@protonmail.com"
__status__ = "Development"
"""

# For the CPU split:
# Type in Terminal (-P*, * must be equal to n_cpu):
#  cat args_splitter | xargs -L1 -P6 python3 JA_Server_Sim.py
# Note: args_splitter must contain numbers from 1 to n_cpu


# TODO: new Fitness calculation: muliplication of each trial, sqrt^4 (+ drop slow trial if possible)
# TODO: Next 50.000 generations., mode2

n_cpu = 6

if len(sys.argv) > 1 and sys.argv[1].isdigit():
    split = int(sys.argv[1]) if int(sys.argv[1]) <= n_cpu else False
else:
    split = False

if not split:  # is False
    audicon = audio_condition_request()
    number_of_generations = generation_request()
    filename = filename_request("joint")

    # If e.g. scalar = 0.336 => Target just makes one turn
    scalar = simlength_scalar_request()
    vary_scalar_mode = scalar_mode_request()

    # Whether sensory and motor weights suppose to be symmetrical:
    symmetry = symmetrical_weights_request()


else:  # if splitter is used, these values must be manually filled, here in python file
    # Manually adjust the following parameters:
    audicon = True          # True or False
    number_of_generations = 25000
    scalar = 1              # 1 == no scaling [Default] == 3 turns, e.g. 1/3 == first turn
    vary_scalar_mode = 3    # 1, 2 or 3.
    #                           1:= no varying of simulation-length;
    #                           2:= scalar will increase with n_generation between [0.33, 1.66] (== [1Turn, 5Turns]);
    #                           3:= scalar will randomly vary between [0.33, 1.66] (== [1Turn, 5Turns]); TODO: see return_scalar() in Formulas.py
    symmetry = False        # True or False
    filename = None         # or "Gen1001-10000.popsize55.mut0.1.sound_cond=False.sym_weights.JA.joint(Fitness7.58)"
    print("Splitter {} started!".format(split))


ja = JA_Evolution(auditory_condition=audicon, pop_size=55, simlength_scalar=scalar, scalar_mode=vary_scalar_mode, symmetrical_weights=symmetry)


if isinstance(filename, str):
    ja.reimplement_population(filename=filename, plot=False)
    if not split or split == n_cpu:
        if audicon != ja.condition:
            print("...")
            print("Note: Initial Sound Condition differs from the one in implemented file!")
            print("...")


# RUN:
if not split or split == n_cpu:
    print("Run Evolution: {} Generations, Joint Condition, Sound Condition={}, simlength-scalar {}, scalar_mode {}, sym. weights={}".format(
        number_of_generations, ja.condition, scalar, vary_scalar_mode, symmetry))
ja.run_evolution(generations=number_of_generations, splitter=split, n_cpu=n_cpu)

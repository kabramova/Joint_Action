from JA_Evolution import *
from SA_Evolution import *


# Setup Agent(s) to analyse:
condition = single_or_joint_request()

audicon = audio_condition_request()
number_of_generations = 2
filename = filename_request(condition)  # "joint" or "single"
# filename = "Gen1001-2000.popsize55.mut0.02.sound_cond=False.JA.joint(Fitness6.1)"

if condition == "single":
    sa = SA_Evolution(auditory_condition=audicon)
    if isinstance(filename, str):
       sa_performance = sa.reimplement_population(filename=filename, Plot=True)
        # sa_performance[0-3] are the different trials
       # ja_performance[0-3][0-5] = fitness[0], trajectories[1], keypress[2], sounds[3], neural_state[4], neural_input_L[5]

if condition == "joint":
    ja = JA_Evolution(auditory_condition=audicon, pop_size=55)
    if isinstance(filename, str):
        ja_performance = ja.reimplement_population(filename=filename, Plot=True)
        # ja_performance[0-3] are the different trials
        # ja_performance[0-3][0-7] = fitness[0], trajectories[1], keypress[2], sounds[3], neural_state_L[4], neural_state_L[5], neural_input_L[6], neural_input_L[7]



len(sa_performance)
sa_performance[3]





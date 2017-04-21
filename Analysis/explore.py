import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

condition = "single"
audicon = False
load = True
lesion = False
lesion_name = ""

sa_performance = np.load('/Users/katja/PycharmProjects/Joint_Action/SimonAnalysis/single/sa_performance_condFalse_fitness6.19.npy')
#loaded sa_performance_condFalse_fitness6.19.npy

perf = sa_performance
# perf.shape
# (4,6) - the rows are the different types of trials, the columns are fitness[0], trajectories[1], keypress[2],
# sounds[3], neural_state[4], neural_input_L[5]

fitness = np.round(np.mean([i[0] for i in sa_performance]), 2)  # Fitness over all trials

# focusing on one trial type, what is recorded is what happened during the whole trial
# perf[0][1].shape; perf[0][2].shape; perf[0][3].shape
# (3635, 2)
# perf[0][4].shape; perf[0][5]
# (3635, 8) - neural state and input for all 8 neurons

# how to access the weights of that evolved agent?

# plotting phase lines

trial = perf[0]
target = trial[1][:,1]
tracker = trial[1][:,0]

df = pd.DataFrame(trial[4], columns=["n1", "n2", "n3", "n4", "n5", "n6", "n7","n8"])
df['target'] = pd.Series(target, index=df.index)

#plt.plot(target, trial[4])

df.set_index('target').plot()
plt.xlim(-20.5, 20.5)
plt.show()

# target - tracker
distance = abs(target - tracker)


# target to border
def get_border_distance(pos):
    if pos >=0:
        return 20-pos
    else:
        return abs(-20-pos)
vfunc = np.vectorize(get_border_distance)
border_distance = vfunc(target)



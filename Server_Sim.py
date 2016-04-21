from Evolution import *


### Simulation of Evolution
e1 = Evolution(simlength=5000)
Fitness_progress, pos_target = e1.run_evolution(Generations=500, mutation_var=.0001, complex_trials=True, fit_prop_sel=True, position_agent=[50,50], angle_to_target= np.pi/2,  distance_to_target = 30)

# e1.reimplement_population(Filename=None, Plot=True) # Filename starts with "sim..."

'''
XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX
XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX
XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX
'''

e2 = Evolution(simlength=5000)
Fitness_progress, pos_target = e2.run_evolution(Generations=500, mutation_var=.02, complex_trials=True, fit_prop_sel=False, position_agent=[50,50], angle_to_target= np.pi/2,  distance_to_target = 30)

# e2.reimplement_population(Filename=None, Plot=True) # Filename starts with "sim..."


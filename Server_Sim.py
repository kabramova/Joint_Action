from Evolution import *


### Simulation of Evolution
# e1 = Evolution(simlength=5000)
# Fitness_progress = e1.run_evolution(Generations=500, mutation_var=.0001, complex_trials=True, fit_prop_sel=True, position_agent=[50,50], angle_to_target= np.pi/2,  distance_to_target = 30)

# e1.reimplement_population(Filename=None, Plot=True) # Filename starts with "sim..."

'''
XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX
XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX
XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX
'''

e2 = Evolution(simlength=7500)
e2.filename = "sim5000.mut0.02.Gen4001-5500_CT=True.fps=False"
e2.reimplement_population(Filename=None, Plot=False)
e2.run_evolution(Generations=1500, mutation_var=.02, complex_trials=True, fit_prop_sel=False, position_agent=[50,50], angle_to_target= np.pi/2,  distance_to_target = 30)


# import os
# os.listdir()
# e2.reimplement_population(Filename=None, Plot=True) # Filename starts with "sim..."
# e2.plot_pop_list(2)
# e2._set_target(complex=True)
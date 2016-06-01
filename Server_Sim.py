from Evolution import *

### Simulation of Evolution

e1 = Evolution(simlength=5000)

filename = ""
if filename != "" and isinstance(filename, str):
    e1.reimplement_population(Filename=filename, Plot=False)
    print("Filename:", e1.filename)
e1.run_evolution(Generations=1000, mutation_var=.02, complex_trials=True, fit_prop_sel=False, position_agent=[50,50], angle_to_target= np.pi/2,  distance_to_target = 30)

# os.listdir()
# e1.reimplement_population(Filename=None, Plot=True) # Filename starts with "sim..."
# e1.plot_pop_list(2)
# e1._set_target(complex=True)


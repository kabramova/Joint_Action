from Evolution import *

### Simulation of Evolution

e1 = Evolution(simlength=5000)

# os.listdir('poplists')
filename = "sim5000.mut0.02.Gen7001-8000(Fitness 24.71)"      # Filename starts with "sim..."
if filename != "" and isinstance(filename, str):
    e1.reimplement_population(Filename=filename, Plot=False)
    print("File implemented:", e1.filename)

n_gen = generation_request()

print("Run Evolution from Generations {}-{}".format(e1.Generation, e1.Generation+n_gen))
e1.run_evolution(Generations=n_gen, mutation_var=.02, complex_trials=True, fit_prop_sel=False, position_agent=[50,50], angle_to_target= np.pi/2,  distance_to_target = 30)


# e1.reimplement_population(Filename=filename, Plot=True)
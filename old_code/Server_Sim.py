from old_code.Evolution import *

"""
__author__  = Simon Hofmann"
__credits__ = ["Simon Hofmann", "Katja Abramova", "Willem Zuidema"]
__version__ = "1.0.1"
__date__ "2016"
__maintainer__ = "Simon Hofmann"
__email__ = "simon.hofmann@protonmail.com"
__status__ = "Development"
"""

# Simulation of Evolution

e1 = Evolution(simlength=5000)

# os.listdir('SimonPoplists')
filename = None  # "sim5000.mut0.02.Gen12001-15000(Fitness 24.45)"      # Filename starts with "sim..."
if filename != "" and isinstance(filename, str):
    e1.reimplement_population(filename=filename, plot=False)
    print("File implemented:", e1.filename)

n_gen = generation_request()

print("Run Evolution from Generations {}-{}".format(e1.generation, e1.generation+n_gen))
e1.run_evolution(generations=n_gen, mutation_var=.25, complex_trials=True, fit_prop_sel=False, position_agent=[50, 50],
                 angle_to_target=np.pi/2, distance_to_target=30)  # mutation rate default 0.01


# Plot current file:
# e1.reimplement_population(filename=filename, plot=True)

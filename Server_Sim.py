from Evolution import *

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

# os.listdir('poplists')
filename = "sim5000.mut0.02.Gen8751-10000(Fitness 24.54)"      # Filename starts with "sim..."
if filename != "" and isinstance(filename, str):
    e1.reimplement_population(filename=filename, plot=False)
    print("File implemented:", e1.filename)

n_gen = generation_request()

print("Run Evolution from Generations {}-{}".format(e1.Generation, e1.Generation+n_gen))
e1.run_evolution(generations=n_gen, mutation_var=.02, complex_trials=True, fit_prop_sel=False, position_agent=[50, 50],
                 angle_to_target=np.pi/2, distance_to_target=30)

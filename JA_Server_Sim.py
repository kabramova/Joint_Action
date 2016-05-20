from JA_Evolution import *

ja = JA_Evolution(auditory_condition=True)
ja.run_evolution(generations=100)

ja2 = JA_Evolution(auditory_condition=False)
ja2.run_evolution(generations=100)

ja.print_best(3)

# TODO: for reimplement check globalisation in particular if run_and_plot
# ja.reimplement_population(filename="sim2789.mut0.02.Gen11-15.JA.single")
# ja.run_evolution(generations=100)

# ja_tables = ja.run_and_plot()


# TODO: own evolution for trials with sound output/input

sim = JA_Simulation(auditory_condition=True)
sim.setup(trial_speed="fast")
tab = sim.run_and_plot()

sim = JA_Simulation(auditory_condition=False)
sim.setup(trial_speed="slow")
tab = sim.run_and_plot()


# Test

ja = JA_Evolution(auditory_condition=True)
ja.run_and_plot()
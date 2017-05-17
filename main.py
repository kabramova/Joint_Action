"""
This is the main file for running evolution of neural network agents in the Knoblich and Jordan (2003) task.
"""
import random
from evolve import Evolution
import json
# from profilestats import profile


# load configuration settings
json_data = open('config.json')
config = json.load(json_data)
json_data.close()


# @profile(print_stats=10)
def main():
    # set random seed
    random.seed(592)

    # set up evolution
    evolution = Evolution(config['evolution_params']['pop_size'],
                          config['evolution_params'],
                          config['network_params'],
                          config['evaluation_params'],
                          config['agent_params'])

    # run evolution
    evolution.run()
    return


if __name__ == '__main__':
    main()

"""
This is the main file for running evolution of neural network agents in the Knoblich and Jordan (2003) task.
This version parallelizes the seeds.
"""
import random
from evolve import Evolution
import json
import argparse
import os
from multiprocessing import Process
import shutil
# from profilestats import profile


# @profile(print_stats=10)
def do_evolution(parent_dir, agent_type, seed_num, mutation_variance, prob_crossover):
    # load configuration settings
    json_data = open('config.json')
    config = json.load(json_data)
    json_data.close()

    # set random seed
    random.seed(seed_num)

    # set up evolution
    evolution = Evolution(config['evolution_params']['pop_size'],
                          config['evolution_params'],
                          config['network_params'],
                          config['evaluation_params'],
                          config['agent_params'])

    evolution.evaluation_params['velocity_control'] = agent_type

    if mutation_variance:
        evolution.evolution_params['mutation_variance'] = mutation_variance
    if prob_crossover:
        evolution.evolution_params['prob_crossover'] = prob_crossover

    # create the right directory
    foldername = parent_dir + '/' + str(seed_num)
    evolution.set_foldername(foldername)
    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(foldername)

    # run evolution from scratch or starting from a given population
    evolution.run(None, parallel_agents=False)
    # evolution.run(150)


if __name__ == '__main__':
    # run with  python simulate.py real > kennylog.txt
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_type", type=str, help="specify the type of the agent you want to run",
                        choices=["buttons", "direct"])
    parser.add_argument("seed_list", nargs='+', type=int)
    parser.add_argument("-m", "--mutation_variance", type=int, default=1, help="specify the mutation variance")
    parser.add_argument("-c", "--prob_crossover", type=int, default=0.8, help="specify the probability of crossover")
    args = parser.parse_args()

    parent_dir = os.getcwd() + '/Agents/' + args.agent_type
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    procs = []

    for seed_num in args.seed_list:
        proc = Process(target=do_evolution, args=(parent_dir, args.agent_type, seed_num, args.mutation_variance, args.prob_crossover,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

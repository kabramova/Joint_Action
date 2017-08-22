from analyze import load_population
import os
import pickle
from agents import EmbodiedAgentV2, DirectVelocityAgent
import json
from copy import deepcopy
from CTRNN import BrainCTRNN


def get_best_agents(agent_type):
    agent_directory = "./Agents/single/" + agent_type
    agent_folders = list(filter(lambda f: not f.startswith('.'), os.listdir(agent_directory)))
    agents = []
    for f in agent_folders:
        seed_files = list(filter(lambda f: f.startswith('gen'),
                                 os.listdir(agent_directory + '/{}'.format(f))))
        gen_numbers = [int(x[3:]) for x in seed_files]
        agents.extend(load_population('single', agent_type, f, max(gen_numbers)))
    agents.sort(key=lambda agent: agent.fitness, reverse=True)
    best_agents = agents[:50]
    return best_agents


json_data = open('config.json')
config = json.load(json_data)
json_data.close()


def reconstruct_agent(agent, agent_type):
    if agent_type == "buttons":
        agent_brain = BrainCTRNN(config['network_params']['num_neurons'],
                                 config['network_params']['step_size'],
                                 config['network_params']['tau_range'],
                                 config['network_params']['g_range'],
                                 config['network_params']['theta_range'],
                                 config['network_params']['w_range'])
        new_agent = EmbodiedAgentV2(agent.brain, config['agent_params'],
                                config['evaluation_params']['screen_width'])
    else:
        new_agent = DirectVelocityAgent(agent.brain, config['agent_params'],
                                config['evaluation_params']['screen_width'])
    new_agent.brain.__dict__ = deepcopy(agent.brain.__dict__)
    new_agent.__dict__ = deepcopy(agent.__dict__)
    return new_agent


def save_best_agents(agent_type):
    best_agents = get_best_agents(agent_type)
    population = []
    for agent in best_agents:
        population.append(reconstruct_agent(agent, agent_type))
    pop_file = open('./Agents/joint/{}/seedpop'.format(agent_type), 'wb')
    pickle.dump(population, pop_file)
    pop_file.close()

save_best_agents("buttons")
save_best_agents("direct")

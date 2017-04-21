import simulate
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import datetime
import os

step_size = 0.01
# evaluation parameters
screen_width = [-20, 20]
velocities = [3.3, 4.3, -3.3, -4.3]
impact = [0.7, 1.0]
condition = "no-sound"

# trials
# [(3.3, 0.7), (3.3, 1.0), (4.3, 0.7), (4.3, 1.0), (-3.3, 0.7), (-3.3, 1.0), (-4.3, 0.7), (-4.3, 1.0)]
v_names = ["slow", "fast", "slow", "fast"]
im_names = ["low", "high"]
trial_names = [x + "." + y for x in v_names for y in im_names]

popfile = open('./Agents/gen430', 'rb')
population = pickle.load(popfile)
popfile.close()

agent = population[1]

simulation_run = simulate.Simulation(screen_width, step_size, velocities, impact)
trial_data = simulation_run.run_trials(agent, simulation_run.trials, savedata=True)

agent.brain.randomize_state([0,0])  # all inner neural states = 0

fitness = trial_data['fitness'][0]
target_pos = trial_data['target_pos'][0]
tracker_pos = trial_data['tracker_pos'][0]
keypress = trial_data['keypress'][0]
sim_length = simulation_run.sim_length[0]

# Save the current state of the system
times = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# Create folder for images
os.makedirs("./Animations/{}".format(times))

ticker = 10  # just plot every 10th (x-th) state.
counter_img = 0
counter_sec = 0

# Region Borders
upper_bound = screen_width[1]
lower_bound = screen_width[0]
screen_size = upper_bound - lower_bound
region_width = screen_size / 3
right_border = round(0 + region_width / 2, 2)
left_border = round(0 - region_width / 2, 2)

# Set initial Target Direction:
if target_pos[0] > 0:
    direction = "right"
else:
    direction = "left"

yrange = range(-5, 5)

for i in range(0, sim_length, ticker):

    plt.figure(figsize=(10, 6), dpi=80)

    plt.plot(np.repeat(left_border, len(yrange)), yrange, "--", c="grey", alpha=0.2)  # Region Left
    plt.plot(np.repeat(right_border, len(yrange)), yrange, "--", c="grey", alpha=0.2)  # Region Right

    plt.plot(tracker_pos[i], 0, 'ro', markersize=12, alpha=0.5)  # Tracker
    plt.plot(target_pos[i], 0, 'go')  # Target

    if any(keypress[i:i+ticker, 0] == -1):
        plt.plot(-10, -4, 'bs', markersize=16)  # keypress left
    if any(keypress[i:i+ticker, 1] == 1):
        plt.plot(10, -4, 'bs', markersize=16)  # keypress right

    # if condition == "sound":
    #     if any(sounds[i:i + ticker, 0] == 1):
    #         plt.plot(-10, -3.9, 'yo', markersize=24, alpha=0.3)  # sound left
    #     if any(sounds[i:i + ticker, 1] == 1):
    #         plt.plot(10, -3.9, 'yo', markersize=24, alpha=0.3)  # sound right

    # Define boarders
    plt.xlim(-20, 20)
    plt.ylim(-5, 5)

    # Remove y-Axis
    plt.yticks([])

    # Print fitnesss, time and conditions in plot
    plt.annotate(xy=[0, 4], xytext=[0, 4], s="Trial Fitness: {}".format(round(fitness, 2)))  # Fitness

    # Updated time-counter:
    if counter_img == 25:
        counter_sec += 1
        print("{}% ready".format(np.round((i / sim_length) * 100, 2)))  # gives feedback how much is plotted already.

    counter_img = counter_img + 1 if counter_img < 25 else 1

    # Update simulation time:
    sim_msec = i if i < 100 else i % 100
    sim_sec = int(i * step_size)  # or int(i/100)

    plt.annotate(xy=[-15, 4], xytext=[-15, 4], s="{}:{}sec (Real Time)".format(str(counter_sec).zfill(2),
                                                                               str(counter_img).zfill(2)))  # Real Time

    plt.annotate(xy=[-15, 3.5], xytext=[-15, 3.5], s="{}:{}sec (Simulation Time)".format(str(sim_sec).zfill(2),
                                                                                         str(sim_msec).zfill(2)))  # Simulation Time

    plt.annotate(xy=[0, 3.5], xytext=[0, 3.5], s="Initial Target Direction: {}".format(direction))  # Target Direction
    plt.annotate(xy=[0, 3.0], xytext=[0, 3.0], s="Trial {}".format(trial_names[0]))  # trial
    plt.annotate(xy=[-15, 3.0], xytext=[-15, 3.0], s="Sound Condition: {}".format(condition))  # condition

    plt.savefig('./Animation/{}/animation{}.png'
                .format(times,
                        str(int(i / ticker)).zfill(len(str(int(sim_length / ticker))))))

    plt.close()

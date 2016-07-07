from Formulas import *
from RollingBot import CatchBot
import matplotlib.pyplot as plt

"""
__author__ = Simon Hofmann"
__credits__ = ["Simon Hofmann", "Katja Abramova", "Willem Zuidema"]
__version__ = "1.0.1"
__date__ "2016"
__maintainer__ = "Simon Hofmann"
__email__ = "simon.hofmann@protonmail.com"
__status__ = "Development"
"""

class Simulate:
    """ ... """

    def __init__(self, simlength=500):
        self.agent = CatchBot()
        self.simlength = simlength

    def run(self):
        self.agent.movement(self.simlength)
        return np.matrix(self.agent.position)

    def run_and_plot(self, colour=None, animation=False):

        pos = np.matrix(self.agent.position)

        for _ in np.arange(self.simlength):
            self.agent.movement()
            pos = np.concatenate((pos, np.matrix(self.agent.position)))

        if animation:

            print("Animate trajectory for Target-Position: {}".format(np.round(self.agent.position_target)))

            # Create folder:
            # times = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            # os.makedirs("./Animation/{}.Animation.Catch_Target_pos_{}".format(times, np.round(self.agent.position_target)))
            folder = "./Animation/Animation.Catch_Target"
            if not os.path.exists(folder):
                os.makedirs(folder)

            ticker = 75       # just plot every ticker-th state.
            counter_img = 0
            counter_sec = 0

            for i in np.arange(0, self.simlength, ticker):

                # For Fast Trials: with a simlength of 2789 the resulting gif-animation is approx. 11sec long (25frames/sec)
                # & for Slow Trials: with a simlength of 3635.
                # we can change the animation length by changing the modulo here [i%x].

                plt.plot(pos[i, 0], pos[i, 1], marker="o", markersize=2, markeredgewidth=0.0, mfc=colour)  # Tracker

                # Updated time-counter:
                if counter_img == 25:
                    counter_sec += 1
                    # print("Time: {} sec".format(str(counter_sec).zfill(2)))  # Time

                counter_img = counter_img + 1 if counter_img < 25 else 1

                plt.annotate(xy=[1, -10], xytext=[1, -10], backgroundcolor="w" ,
                             s="Time = {}:{}sec".format(str(counter_sec).zfill(2),
                                                        str(counter_img).zfill(2)))  # Time

                j = 0
                while os.path.exists("{}/animation{}.png".format(folder, str(j).zfill(3))):
                    j += 1
                plt.savefig('{}/animation{}.png'.format(folder, str(j).zfill(3)))

        # Plot
        if colour is None:
            plt.plot(pos[:, 0], pos[:, 1])
            plt.plot(pos[-1, 0], pos[-1, 1], 'ro', alpha=.5)
        else:
            plt.plot(pos[:, 0], pos[:, 1], c=colour)
            plt.plot(pos[-1, 0], pos[-1, 1], marker="o", alpha=0.5, c=colour)

            # plt.plot(self.agent.position_target[0], self.agent.position_target[1], 'ro')
            # plt.axis([0, 100, 0, 100])

# Simulate(1000).run_and_plot()

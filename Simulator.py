from Formulas import *
from RollingBot import CatchBot
import matplotlib.pyplot as plt


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

        if colour is None:
            plt.plot(pos[:, 0], pos[:, 1])
            plt.plot(pos[-1, 0], pos[-1, 1], 'ro', alpha=.5)
        else:
            plt.plot(pos[:, 0], pos[:, 1], c=colour)
            plt.plot(pos[-1, 0], pos[-1, 1], marker="o", alpha=0.5, c=colour)

            # plt.plot(self.agent.position_target[0], self.agent.position_target[1], 'ro')
            # plt.axis([0, 100, 0, 100])

        # TODO: Animate:
        # if animation:
        #     # Create folder:
        #     times = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        #     os.makedirs("./Animation/{}.Animation.Catch_Target_pos_{}".format(times, self.agent.position_target))


# Simulate(1000).run_and_plot()

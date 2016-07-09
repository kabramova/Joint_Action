from Formulas import *

"""
__author__  = Simon Hofmann"
__credits__ = ["Simon Hofmann", "Katja Abramova", "Willem Zuidema"]
__version__ = "1.0.1"
__date__ "2016"
__maintainer__ = "Simon Hofmann"
__email__ = "simon.hofmann@protonmail.com"
__status__ = "Development"
"""


class CTRNN:

    def __init__(self, number_of_neurons, timestep=0.01):

        """
        :param number_of_neurons:
        :param timestep:
        """"""
        Initialize whole random fully connected CTRNN of size N (number_of_neurons)
        with parameter vectors:

        Y      = 'state of each neuron' at current timepoint i
        Tau    = 'time constant (τ > 0)'
        W     = 'fixed strength of the connection from jth to ith neuron', Weight Matrix
        Theta = 'θ is the bias term'
        sigma = 'σ(x) is the sigmoid function / standard logistic activation function' 1/(1+np.exp(-x))
        I     = 'constant external input' at current timepoint i
        G     = 'gain' (makes neurons highly sensitive to their input, primarily for motor or sensory nodes)
                 Preferably g ∈ [1,5] & just > 1 for neurons connected to sensory input or motor output.

        >>Note: Parameter_ranges all according to Agmon, Beer (2013)
        >>Note: Courant-Friedrichs-Lewy Condition
        :return: output
        """

        self.h = timestep

        self.N = number_of_neurons

        self.Y = np.matrix(np.zeros((self.N, 1)))

        self.TAU_RANGE = [1, 10]
        self.Tau = randrange(self.TAU_RANGE, self.N, 1)

        self.T_RANGE = [-13, 13]
        self.Theta = randrange(self.T_RANGE, self.N, 1)

        # G_RANGE = [1.0, 5.0]
        self.G = np.ones((self.N, 1))

        self.W_RANGE = [-13, 13]
        self.W = randrange(self.W_RANGE, self.N, self.N)

        self.I = np.matrix(np.zeros((self.N, 1)))

    def next_state(self):  # or (self, input, h)
        # compute the next state of the network, using its current state
        # and the euler equations
        # first compute activation motor neurons
        # act_motor = self.compute_motor_neurons()

        o = sigmoid(np.multiply(self.G, self.Y + self.Theta))

        dydt = np.multiply(1 / self.Tau, - self.Y + np.dot(self.W, o) + self.I) * self.h

        self.Y += dydt

'''

for i in range(1):  # how many times to plot
    n1 = CTRNN(8)   # network size

    state_matrixY = n1.Y
    for i in np.arange(1,5000):
        n1.next_state()
        state_matrixY = np.c_[state_matrixY, n1.Y]

    plt.plot(np.arange(1,5000) ,state_matrixY[:,:-1].T)
    plt.title((r'$\frac{dy}{dt} = \frac{1}{\tau}(-y_i + \sum_{j=1}^N w_{ji}\sigma(y_j + \theta_j) + I_i)$' + r'$ , N = {}'.format(n1.N)))

print("Tau:\n", np.round(n1.Tau,2),
      "\n \n weights:\n", np.round(n1.W,2),
      "\n \n Biases:\n", np.round(n1.Theta,2),
      "\n \n last State y:\n", np.round(n1.Y,2))

'''
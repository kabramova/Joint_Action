import numpy as np
import copy
import sys
import os, datetime, time
from collections import OrderedDict
from functools import wraps


"""
__author__  = Simon Hofmann"
__credits__ = ["Simon Hofmann", "Katja Abramova", "Willem Zuidema"]
__version__ = "1.0.1"
__date__ "2016"
__maintainer__ = "Simon Hofmann"
__email__ = "simon.hofmann@protonmail.com"
__status__ = "Development"
"""


# Euler Method:
#        Δy/Δt ≈ dy/dt = f(y,t)
#   =>  Δy ≈ f(y,t)Δt  , where Δt = h

# >>Note: Courant-Friedrichs-Lewy Condition


def sigmoid(x):
    """
    plotting the sigmoid
    x = np.zeros(1)
    for i,j in enumerate(np.arange(-5,5,.001)):
        x = np.c_[x, f.sigmoid(j)]
    plt.plot(np.arange(-5,5,.001),x.transpose()[:-1])
    """
    return 1 / (1 + np.exp(-x))
    # Initialize parameter vectors Y, Tau, Theta, W and I


def randrange(ranges, dimension_1, dimension_2):
    """
    Creates matrix(dim1,dim2) with random numbers in the range of input.
    (b - a) * random_sample() + a, b>a
    """
    return np.matrix((ranges[1] - ranges[0]) * np.random.sample((dimension_1, dimension_2)) + ranges[0])


def vec_angle(v1, v2):
    """
    Computes the minimal positive angle between the two input vectors.
    Vectors will be normalized within the method
    Order of vectors does not play a role.
    :param v1: first vector
    :param v2: second vector
    :return: angle in [0,π)
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))


def vec_angle2(v1, v2):
    """
    Computes the minimal signed(!) angle between the two input vectors.
    Vectors will be normalized within the method
    Positive sign denotes a anti-clockwise rotation.
    Order of vectors does play a role.
    :param v1: first vector
    :param v2: seceond vector
    :return: angle in [-π,π]
    """
    n1 = v1/np.linalg.norm(v1)
    n2 = v2/np.linalg.norm(v2)
    angle = np.arctan2(n2[1], n2[0]) - np.arctan2(n1[1], n1[0])
    if angle < -np.pi:
        angle += np.pi*2
    elif angle > np.pi:
        angle -= np.pi*2
    return angle


def rotate90(vector, clockwise=True):
    """
    Rotates the input vector 90 degrees
    :return: rotated vector
    """
    swap = np.array([vector[1], vector[0]])
    if clockwise:
        rotated_vec = swap * np.array([1, -1])
    else:
        rotated_vec = swap * np.array([-1, 1])

    return rotated_vec


def mat_sort(matrix, index=0, axis=2):
    """
    Sorts Matrix with respect to particular column or row
    :param matrix: Matrix to be sorted
    :param axis: axis-1 = rows; axis-2 = columns
    :param index: index of row/column, which is the "sorting-ground"
    :return: sorted matrix
    """
    if axis == 2:
        return matrix[np.argsort(matrix[:, index])]
    elif axis == 1:
        tmatrix = np.transpose(matrix)
        sort_tmatrix = tmatrix[np.argsort(tmatrix[:, index])]
        return np.transpose(sort_tmatrix)
    else:
        raise ValueError("axis must be either 1 or 2")


def normalize(array):
    """
    :return: rescaled array in range [0,1]
    """
    if np.all(np.array(array) == 0):
        print("Input = zero.array >> no Normalization")
    else:
        return (array - np.min(array)) / (np.max(array) - np.min(array))


# Formulas for JointAction
def angle_velo(beta=None, b=None):
    """
    TRIANGLE:
    a:     = 80cm distance to screen
    b:       velocity in cm of tracker/target
    c:       just needed for calculation
    alpha:   just needed for calculation
    beta:    angle-velocity of tracker/target
    gamma: = 90 degrees (screen is orthogonal to viewer)
    Source:  "http://www.arndt-bruenner.de/mathe/scripts/Dreiecksberechnung.htm"
    :param:   beta, b: (see above)
    :return: b or beta, respectively
    """

    gamma = 90
    a = 80

    if beta is None and b is None:
        raise ValueError("Need input, either for beta or b")

    elif b is None:
        alpha = 180 - beta - gamma
        b = a * np.sin(np.radians(beta)) / np.sin(np.radians(alpha))
        # c = a * np.sin(gamma) / np.sin(alpha)
        return b

    else:  # beta =is None:
        c = np.sqrt(a * a + b * b - 2 * a * b * np.cos(np.radians(gamma)))
        # alpha = np.arccos((a * a - b * b - c * c) / (-2 * b * c))
        beta = np.arccos((b * b - c * c - a * a) / (-2 * c * a))
        return np.degrees(beta)


def angle_velo2(velocity_deg=None, velocity_cm=None):
    """
    TRIANGLE:
    d:     = 80cm distance to screen
    theta:   Angle
    vel_cm:  velocity in cm of tracker/target
    vel_deg: velocity in degrees of tracker/target
    Source:  "http://dragon.uml.edu/psych/eyeview.html"
             "http://www.yorku.ca/eye/visangle.htm"
    :param   velocity_cm, velocity_deg: (see above)
    :return: velocity (cm, degrees, respectively)
    """

    d = 80

    if velocity_deg is None and velocity_cm is None:
        raise ValueError("Need input, either velocity")

    elif velocity_deg is None:
        # vel_deg = np.degrees(np.arctan(velocity_cm / d))   # if, the same as angle_velo (Version1)
        vel_deg = np.degrees(2*np.arctan(velocity_cm/(d*2)))
        return vel_deg

    else:  # velocity_cm is None:
        theta = velocity_deg
        # vel_cm = np.tan(np.radians(theta)) * d # if, the same as angle_velo (Version1)
        vel_cm = 2*np.tan(np.radians(theta/2)) * d
        return vel_cm


# Requests:
def save_request():
    """
    Ask whether results should be saved in external file
    take output of save_request() as follows:
    save = save_request()
    :return True or False
    """

    count = 0
    while count != 3:
        inputs = input("Do you want to save the final population ('(y)es'/'(n)o'):")

        if inputs in ["y", "Y", "yes", "Yes", "YES"]:
            print("Saving final population in external file")
            return True

        elif inputs in ["n", "N", "no", "No", "NO"]:
            print("Final population won't be saved")
            return False

        else:
            print("Input is not understood.\n"
                  "Type either 'yes' or 'no'.\n"
                  "{} more attempt(s)".format(2 - count))
            count += 1

    raise ValueError("Function stopped")


def audio_condition_request():

    condition = input("Sound condition is '(T)rue' / '(F)alse':")

    if condition == 1 or condition.lower() == "t" or condition.lower() == "true":
        condition = True

    elif condition == 0 or condition.lower() == "f" or condition.lower() == "false":
        condition = False

    else:
        raise ValueError("Input must be True or False")

    return condition


def symmetrical_weights_request():

    symmetrical_weights = input("Shall the sensory and auditory weights be symmetrical '(T)rue' / '(F)alse':")

    if symmetrical_weights == 1 or symmetrical_weights.lower() == "t" or symmetrical_weights.lower() == "true":
        symmetry = True
        print("Sensory and auditory weights will be symmetrical")

    elif symmetrical_weights == 0 or symmetrical_weights.lower() == "f" or symmetrical_weights.lower() == "false":
        symmetry = False
        print("Sensory and auditory weights won't be held symmetrical")

    else:
        raise ValueError("Input must be True or False")

    return symmetry


def generation_request():

    number_of_generations = input("How many Generations to run (int):")
    if int(number_of_generations):
        number_of_generations = int(number_of_generations)

    if number_of_generations <= 1:
        raise ValueError("Evolution must run for at least 2 Generations")

    return number_of_generations


def simlength_scalar_request():

    simlength_scalar = input("With what factor you want to scale the simulation length (int) [default=1]:")
    if int(simlength_scalar):
        simlength_scalar = int(simlength_scalar)

    if simlength_scalar <= 0:
        raise ValueError("Scalar must be greater than zero")

    return simlength_scalar


def scalar_mode_request():
    print("Choose between the following scalar_modes:")
    print("")
    print("     1:= no varying of simulation-length;")
    print("     2:= scalar will increase with n_generation between [0.33, 2.0] (== [1Turn, 6Turns]);")
    print("     3:= scalar will randomly vary between [0.33, 2.0] (== [1Turn, 6Turns]);")
    print("")
    scalar_mode = input("Scaler Mode 1, 2 or 3 [default=1]:")

    if int(scalar_mode):
        scalar_mode = int(scalar_mode)

    if scalar_mode < 1 or scalar_mode > 3:
        raise ValueError("Scalar must be 1, 2 or 3")

    return scalar_mode


def return_scalar(scalar_mode=1, current_generation=None, max_generation=None, given_scalar=None, splitter=False):
        """
        Depending on the scalar_mode this returns different scalars to be applied.
        :param scalar_mode: either 1, 2 or 3
        :param current_generation: for scalar_mode 2
        :param max_generation: for scalar_mode 2
        :param given_scalar: for scalar_mode 1
        :return: scalar
        """
        if scalar_mode != 1 and scalar_mode != 2 and scalar_mode != 3:
            raise ValueError("Scalar must be 1, 2 or 3")

        if scalar_mode == 1:
            if not given_scalar:
                raise ValueError("given_scalar must be given")
            scalar = given_scalar

        elif scalar_mode == 2:
            if not current_generation and not max_generation:
                raise ValueError("scalar_modes 2 & 3 need current_generation and max_generation")
            low = 1 / 3.0  # 0.33
            high = 5 / 3.0  # 1.66
            way_length = high - low
            pos = current_generation / float(max_generation)

            scalar = low + way_length * pos

        else:  # scalar_mode == 3:
            # TODO: fitness must be normalized (comparablity between generations/simlengths)
            if not splitter:
                scalar = np.random.uniform(low=1/3.0, high=1 + 2/3.0)  # uniform-distribution between [0.33, 1.66]
                # scalar = np.random.normal(loc=1.0, scale=0.25)  # normal-dist. loc==mean =1.0, scale==sd =0.25 =>> main curve between [0.4, 1.6]

            else:  # splitter is used
                # Fixed random, uniform list of 100 simlengths
                # simlengths_100 = np.linspace(start=1/3.0, stop=1 + 2 / 3.0, num=100)
                # np.random.shuffle(simlengths_100)
                simlengths_100 = np.array([0.42760943, 1.22222222, 1.3973064, 0.62962963, 0.46801347,
                                           1.04713805, 1.31649832, 0.49494949, 1.02020202, 0.81818182,
                                           0.75084175, 1.28956229, 1.14141414, 1.55892256, 1.43771044,
                                           1.00673401, 0.61616162, 1.18181818, 1.34343434, 0.97979798,
                                           0.72390572, 0.52188552, 1.1952862, 1.12794613, 0.34680135,
                                           0.73737374, 0.58922559, 1.07407407, 1.27609428, 0.50841751,
                                           0.71043771, 1.16835017, 1.54545455, 0.95286195, 1.46464646,
                                           1.23569024, 1.20875421, 1.24915825, 0.87205387, 0.33333333,
                                           1.57239057, 0.92592593, 1.15488215, 1.41077441, 1.47811448,
                                           0.57575758, 1.51851852, 1.65319865, 0.54882155, 0.67003367,
                                           1.50505051, 0.44107744, 1.06060606, 0.77777778, 0.79124579,
                                           0.53535354, 0.93939394, 0.76430976, 0.45454545, 1.61279461,
                                           1.5993266, 0.88552189, 0.99326599, 0.4006734, 1.38383838,
                                           1.49158249, 1.53198653, 0.37373737, 0.96632997, 0.64309764,
                                           1.26262626, 0.36026936, 1.42424242, 1.32996633, 0.48148148,
                                           1.45117845, 1.58585859, 1.03367003, 0.8047138, 0.85858586,
                                           0.68350168, 1.08754209, 1.1010101, 0.83164983, 0.8989899,
                                           1.37037037, 1.63973064, 0.6026936, 1.35690236, 0.6969697,
                                           0.65656566, 0.84511785, 1.62626263, 1.66666667, 1.3030303,
                                           0.38720539, 0.91245791, 0.56228956, 1.11447811, 0.41414141])

                # Take value out of simlengths_100 depending on current_generation
                idx = current_generation % 100
                scalar = simlengths_100[idx]

        # Last round of simulation is with 3-Target-Turns for comparability
        if current_generation == max_generation-1:
            scalar = 1.0

        return scalar


def filename_request(single_or_joint):
    found = 0

    assert single_or_joint in ["single", "joint"], 'Wrong input: Either "single" or "joint"'

    request = input("Do you want to implement a file ((y)es, (n)o:")

    if request in ["y", "Y", "yes", "Yes", "YES"]:

        for file in os.listdir('SimonPoplists/{}/'.format(single_or_joint)):
            if file.find("sound") != -1 and file.find(single_or_joint) != -1:
                count = 0
                found += 1
                no = False

                filename = file[file.find("Gen"):]

                while count != 3 and no is False:
                    file_request = input("{} \n Do you want to implement this file ((y)es, (n)o:".format(filename))

                    if file_request in ["y", "Y", "yes", "Yes", "YES"]:
                        print(">> File will be implemented")
                        return filename

                    elif file_request in ["n", "N", "no", "No", "NO"]:
                        print(">> Looking for further files")
                        no = True

                    else:
                        print("Input is not understood.\n"
                              "Type either 'yes' or 'no' ({} more attempt(s))".format(2 - count))
                        count += 1
                        if count >= 3:
                            raise ValueError("Function stopped, input is not understood")

        print("There were {} file(s) to implement".format(found))
        if found > 0:
            print("No file was selected \n".format(found))
            print(">> New evolution will be started")

    else:
        print(">> New evolution will be started")


def single_or_joint_request():

    single_or_joint = input("SimonAnalysis for '(s)ingle' or '(j)oint':")

    if single_or_joint.find("S") != -1 or single_or_joint.find("s") != -1:
        output = "single"
    elif single_or_joint.find("J") != -1 or single_or_joint.find("j") != -1:
        output = "joint"
    else:
        raise ValueError("Input must be either '(s)ingle' or '(j)oint'")

    print("Condition is:", output)
    return output


def load_request():

    count = 0
    while count != 3:
        inputs = input("Do you want to load a performance file ('(y)es'/'(n)o'):")

        if inputs in ["y", "Y", "yes", "Yes", "YES"]:
            print("Loading a performance file")
            return True

        elif inputs in ["n", "N", "no", "No", "NO"]:
            print("Existing Poplist(s) will be evaluated and saved in performance file")
            return False

        else:
            print("Input is not understood.\n"
                  "Type either 'yes' or 'no'.\n"
                  "{} more attempt(s)".format(2 - count))
            count += 1

    raise ValueError("Function stopped")


def load_file(single_or_joint, audio_condition):

    assert single_or_joint in ["single", "joint"], \
        "Input not understood. single_or_joint must be either 'single' or 'joint'!"
    assert audio_condition in [True, False], "Input not understood. audio_condition must be either True or False "

    found = 0

    for file in os.listdir("./SimonAnalysis/{}/".format(single_or_joint)):
        if file.find("performance") != -1 and file.find(str(audio_condition)) != -1:
            count = 0
            found += 1
            no = False

            filename = file[file.find("sa"):] if file.find("sa") != -1 else file[file.find("ja"):]

            while count != 3 and no is False:
                file_request = input("{} \n Do you want to load this file ((y)es, (n)o:".format(filename))

                if file_request in ["y", "Y", "yes", "Yes", "YES"]:
                    print(">> File will be loaded")
                    return np.load("./SimonAnalysis/{}/{}".format(single_or_joint, filename))

                elif file_request in ["n", "N", "no", "No", "NO"]:
                    print(">> Looking for further files")
                    no = True

                else:
                    print("Input is not understood.\n"
                          "Type either 'yes' or 'no' ({} more attempt(s))".format(2 - count))
                    count += 1
                    if count >= 3:
                        raise ValueError("Function stopped, input is not understood")

    print("There were {} file(s) to load".format(found))
    if found > 0:
        print("No file was selected \n".format(found))

    raise ValueError("Evaluate recent poplist(s)")


def animation_request():

    animation = input("Do you want to animate '(T)rue' / '(F)alse':")

    if animation == 1 or animation.lower() == "t" or animation.lower() == "true":
        animation = True

    elif animation == 0 or animation.lower() == "f" or animation.lower() == "false":
        animation = False

    else:
        raise ValueError("Input must be True or False")

    return animation


def lesion_request():
    """
    Ask whether a network should be lesioned
    :return True or False
    """

    count = 0
    while count != 3:
        inputs = input("Do you want to lesion the network ('(y)es'/'(n)o'):")

        if inputs in ["y", "Y", "yes", "Yes", "YES"]:
            print("Network will be lesioned")
            return True

        elif inputs in ["n", "N", "no", "No", "NO"]:
            print("Network stays intact through the whole trial")
            return False

        else:
            print("Input is not understood.\n"
                  "Type either 'yes' or 'no'.\n"
                  "{} more attempt(s)".format(2 - count))
            count += 1

    raise ValueError("Function stopped")


# Timers:
def function_timer(function):
    """ Time and execute input-function """
    start_timer = datetime.datetime.now()

    output = function()  # == function()

    # Estimate Duration of Evolution
    end_timer = datetime.datetime.now()
    duration = end_timer - start_timer
    print("Processing time of {}: {} [h:m:s:ms]".format(function.__name__, duration))

    return output


# Alternative:
def function_timed(function):
    """
    This allows to define new function with the timer-wrapper
    Write:
        @function_timed
        def foo():
            print("Any Function")
    And try:
        foo()
    http://stackoverflow.com/questions/2245161/how-to-measure-execution-time-of-functions-automatically-in-python
    """

    @wraps(function)
    def wrapper(*args, **kwds):
        start_timer = datetime.datetime.now()

        output = function(*args, **kwds)  # == function()

        duration = datetime.datetime.now() - start_timer

        print("Processing time of {}: {} [h:m:s:ms]".format(function.__name__, duration))

        return output

    return wrapper

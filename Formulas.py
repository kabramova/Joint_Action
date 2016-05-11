import numpy as np
import matplotlib.pyplot as plt   # for other scripts
import copy

## Euler Method:
#        Δy/Δt ≈ dy/dt = f(y,t)
##   =>  Δy ≈ f(y,t)Δt  , where Δt = h

# >>Note: Courant-Friedrichs-Lewy Condition

def sigmoid(x):
    ''' plotting the sigmoid
    x = np.zeros(1)
    for i,j in enumerate(np.arange(-5,5,.001)):
        x = np.c_[x, f.sigmoid(j)]
    plt.plot(np.arange(-5,5,.001),x.transpose()[:-1])
    '''
    return 1 / (1 + np.exp(-x))
    # Initialize parameter vectors Y, Tau, Theta, W and I


def randrange(range, dimension_1, dimension_2):
    '''
    Creates matrix(dim1,dim2) with random numbers in the range of input.
    (b - a) * random_sample() + a, b>a
    '''
    return np.matrix((range[1] - range[0]) * np.random.sample((dimension_1, dimension_2)) + range[0])


def vec_angle(v1, v2):
    '''
    Computes the minimal positive angle between the two input vectors.
    Vectors will be normalized within the method
    Order of vectors does not play a role.
    :param v1: first vector
    :param v2: second vector
    :return: angle in [0,π)
    '''
    return np.arccos( np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)) )

def vec_angle2(v1,v2):
    '''
    Computes the minimal signed(!) angle between the two input vectors.
    Vectors will be normalized within the method
    Positive sign denotes a anti-clockwise rotation.
    Order of vectors does play a role.
    :param v1: first vector
    :param v2: seceond vector
    :return: angle in [-π,π]
    '''
    n1 = v1/np.linalg.norm(v1)
    n2 = v2/np.linalg.norm(v2)
    angle = np.arctan2(n2[1], n2[0]) - np.arctan2(n1[1], n1[0])
    if angle < -np.pi:
        angle += np.pi*2
    elif angle > np.pi:
        angle -= np.pi*2
    return angle


def rotate90(vector, clockwise = True):
    '''
    Rotates the input vector 90 degrees
    :param vector:
    :return: rotated vector
    '''
    swap = np.array([vector[1], vector[0]])
    if clockwise:
        rotated_vec = swap * np.array([1,-1])
    else:
        rotated_vec = swap * np.array([-1,1])

    return rotated_vec


def mat_sort(matrix, index = 0, axis = 2):
    '''
    Sorts Matrix with respect to particular column or row
    :param matrix: Matrix to be sorted
    :param axis: axis-1 = rows; axis-2 = columns
    :param index: index of row/column, which is the "sorting-ground"
    :return: sorted matrix
    '''
    if axis == 2:
        return matrix[np.argsort(matrix[:,index])]
    elif axis == 1:
        tmatrix = np.transpose(matrix)
        sort_tmatrix =  tmatrix[np.argsort(tmatrix[:,index])]
        return np.transpose(sort_tmatrix)
    else:
        raise ValueError("axis must be either 1 or 2")


def normalize(array):
    '''
    :return: rescaled array in range [0,1]
    '''
    if np.all(np.array(array) == 0):
        print("Input = zero.array >> no Normalization")
    else:
        return (array - np.min(array)) / (np.max(array) - np.min(array))



## Formulas for JointAction

def angle_velo(beta = None, b = None):
    '''
    TRIANGLE:
    a:     = 80cm distance to screen
    b:       velocity in cm of tracker/target
    c:       just needed for calculation
    alpha:   just needed for calculation
    beta:    angle-velocity of tracker/target
    gamma: = 90 degrees (screen is orthogonal to viewer)
    Source:  "http://www.arndt-bruenner.de/mathe/scripts/Dreiecksberechnung.htm"
    :param   beta, b: (see above)
    :return: b or beta, respectively
    '''

    gamma = 90
    a = 80

    if beta == None and b == None:
        raise ValueError("Need input, either for beta or b")

    elif b==None:
        alpha = 180 - beta - gamma
        b = a * np.sin(np.radians(beta)) / np.sin(np.radians(alpha))
        # c = a * np.sin(gamma) / np.sin(alpha)
        return b

    else: # beta == None:
        c = np.sqrt(a * a + b * b - 2 * a * b * np.cos(np.radians(gamma)))
        # alpha = np.arccos((a * a - b * b - c * c) / (-2 * b * c))
        beta = np.arccos((b * b - c * c - a * a) / (-2 * c * a))
        return np.degrees(beta)


def angle_velo2(velocity_deg = None, velocity_cm = None):
    '''
    TRIANGLE:
    d:     = 80cm distance to screen
    theta:   Angle
    vel_cm:  velocity in cm of tracker/target
    vel_deg: velocity in degrees of tracker/target
    Source:  "http://dragon.uml.edu/psych/eyeview.html"
             "http://www.yorku.ca/eye/visangle.htm"
    :param   velocity_cm, velocity_deg: (see above)
    :return: velocity (cm, degrees, respectively)
    '''

    d = 80

    if velocity_deg == None and velocity_cm == None:
        raise ValueError("Need input, either velocity")

    elif velocity_deg==None:
        # vel_deg = np.degrees(np.arctan(velocity_cm / d))   # if, the same as angle_velo (Version1)
        vel_deg = np.degrees(2*np.arctan(velocity_cm/(d*2)))
        return vel_deg

    else: # velocity_cm == None:
        theta = velocity_deg
        # vel_cm = np.tan(np.radians(theta)) * d # if, the same as angle_velo (Version1)
        vel_cm = 2*np.tan(np.radians(theta/2)) * d
        return vel_cm


def save_request():
    '''
    Ask whether results should be saved in external file
    take output of save_request() as follows:
    save = save_request()
    :return True or False
    '''
    count = 0
    while count != 3:
        Input = input("Do you want to save the final population ('(y)es'/'(n)o'):")

        if Input in ["y", "Y", "yes", "Yes", "YES"]:
            print("Saving final population in external file")
            return True

        elif Input in ["n", "N", "no", "No", "NO"]:
            print("Final population won't be saved")
            return False

        else:
            print("Input is not understood.\n"
                  "Type either 'yes' or 'no'.\n"
                  "{} more attempt(s)".format(2 - count))
            count += 1

    raise ValueError("Function stopped")
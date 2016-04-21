import numpy as np
import matplotlib.pyplot as plt   # for other scripts

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
    if array != 0:
        return (array - np.min(array)) / (np.max(array) - np.min(array))
    else:
        print("Input = 0, No Normalization")




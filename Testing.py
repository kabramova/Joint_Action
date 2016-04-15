from Formulas import *
from RollingBot import CatchBot
# from Evolution import Evolution


### Study angles between Target and agent_positions:
Target = np.array([3,3])

# different positions of an agent
a = np.array([3,5])
b = np.array([5,3])
c = np.array([5,2])
d = np.array([3,1])
e = np.array([2,2])
f = np.array([1,3])


positions = np.array([a, b, c, d, e, f])

## 5 different orientations
w1 = np.pi/2
w2 = np.pi * 2/3
w3 = np.pi
w4 = np.pi * 7/4
w5 = - np.pi * 3/4  # =  np.pi * 5/4

winkels = np.array([w1, w2, w3, w4, w5])

winkel_between = np.zeros((np.size(winkels), np.shape(positions)[0] ))

for j,k in enumerate(positions):

    for n,i in enumerate(winkels):
        wCoord = np.array([np.cos(i), np.sin(i)])

        wT = Target - k  # vector between Target and agent

        # nwT = wT / np.linalg.norm((wT)) # normalized, not necessary because its already in method vec_angle(v1,v2)

        winkel_between[n,j] = vec_angle(wCoord,wT) # np.around(np.degrees()) # ... otherwise here nwT instead of wT

print(winkel_between)


### Eye position

# Angles agent:
w1 = np.pi/2        # looks up
w2 = np.pi * 3/2    # looks down
w3 = 0              # looks right
w4 = np.pi          # looks left
w5 = np.pi/4        # looks up-right
w6 = np.pi * 5/4    # looks down-left

angles = np.array([w1, w2, w3, w4, w5, w6])

pos0 = np.array([0,0])
pos1 = np.array([1,1])
pos2 = np.array([1,-1])
pos3 = np.array([-1,-1])

position = np.array([pos0, pos1, pos2, pos3])

for pos in position:
    Eyes = np.zeros((np.size(angles),4))
    for i,w in enumerate(angles):

        angle = np.array([np.cos(w), np.sin(w)])
        rotangle = rotate90(angle, clockwise=True)

        # print(np.dot(angle,rotangle))  # = 0, orthogonal

        eyeL = pos + (-.02) * rotangle
        eyeR = pos + (+.02) * rotangle

        np.concatenate((eyeL,eyeR))

        Eyes[i] = np.concatenate((eyeL,eyeR))

    print(pos)
    print(np.around(Eyes,3))


## How to avoid big self.Angle's:
'''
oldAngle = self.Angle
self.Angle = oldAngle + Torque * self.h
'''

alpha = 2926.9932997073547 # this is e.g. a self.angle after 1000 iterations
alpha_orientation = np.array([np.cos(alpha), np.sin(alpha)]) # this would be the corresponding orientation of the agent

beta = np.mod(alpha,2*np.pi)   # with modulo we can scale it down again
beta_orientation = np.array([np.cos(beta), np.sin(beta)])

print(alpha_orientation)
print(beta_orientation)

np.around(alpha_orientation,10) == np.around(beta_orientation,10)

delta = np.mod(beta,2*np.pi)

beta == delta


## -/+ Angle should indicate whether Target is in clockwise or anti-clockwise direction (angle)
# in CatchBot > def visual_input(self) > print(angle_target_eyes) added

bot1 = CatchBot()
bot2 = CatchBot()
bot3 = CatchBot()
bot4 = CatchBot()
bot5 = CatchBot()
bot6 = CatchBot()
bot7 = CatchBot()
bot8 = CatchBot()

bot1.Angle  = np.pi/4 # looks to the top-right
bot1.position = np.array([50,80]) # (target bottom) = -45°

bot2.Angle  = np.pi/4
bot2.position = np.array([60,80]) # (target bottom-right) = -90°

bot3.Angle  = np.pi/4 # looks to the top-right
bot3.position = np.array([70,80]) # (target bottom-right+) = - 108°

bot4.Angle  = 3*np.pi/4 # looks to the top-left
bot4.position = np.array([50,80]) # (target bottom) = 45°

bot5.Angle  = 3*np.pi/4
bot5.position = np.array([40,80]) # (target bottom-left)= 90°

bot6.Angle  = 3*np.pi/4
bot6.position = np.array([30,80]) # (target bottom-left+) = 108°

bot7.Angle  = 3*np.pi/4
bot7.position = np.array([60,80]) # (target bottom-right) = 0°

bot8.Angle  = np.pi/2 # looks up
bot8.position = np.array([50,10]) # (target bottom+) = 0°

# Angles of left and right eye: left.eye-right.eye >/=/< 0 suggest whether agent looks right or left to target:
bot1.visual_input() # < 0 : 45 left(-)
bot2.visual_input() # = 0 : 90 left(-) at 90° sign changes
bot3.visual_input() # > 0 : 108 left(-)
bot4.visual_input() # > 0 : 45 right(+)
bot5.visual_input() # = 0 : 90 right(+) at 90° sign changes
bot6.visual_input() # < 0 : 108 right(+)
bot7.visual_input() # = 0 : 0 straigth(+/-) at 0°
bot8.visual_input() # = 0 : 0 straigth(+/-)

# => network should be able to calculate the direction correctly without the sign of the angle (left/right).


def vec_angle2(v1,v2):
    '''
    Computes the minimal signed(!) angle between the two input vectors.
    Vectors will be normalized within the method
    Order of vectors does play a role. Positive sign denotes a anti-clockwise rotation
    :param v1: first vector
    :param v2: seceond vector
    :return: angle in (-π,π)
    '''
    n1 = v1/np.linalg.norm(v1)
    n2 = v2/np.linalg.norm(v2)
    angle = np.arctan2(n2[1], n2[0]) - np.arctan2(n1[1], n1[0])
    if angle < -np.pi:
        angle += np.pi*2
    elif angle > np.pi:
        angle -= np.pi*2
    return angle

# 1. Quadrant
a = np.array([1,1])
b = np.array([-1,1])
c = np.array([-2,1])
d = np.array([-1,-1])
e = np.array([-.5,-1])

np.degrees(vec_angle2(a,b))     #  90 ok
np.degrees(vec_angle2(b,a))     # -90 ok
np.degrees(vec_angle2(a,c))     #  108 ok
np.degrees(vec_angle2(c,a))     # -108 ok
np.degrees(vec_angle2(a,d))     # -180 , could be also 180   (ok)
np.degrees(vec_angle2(d,a))     #  180 , could be also -180  (ok)
np.degrees(vec_angle2(a,e))     # -161 ok
np.degrees(vec_angle2(e,a))     #  161 ok

# 2. Quadrant
a = np.array([-1,1])
b = np.array([-1,-1])

np.degrees(vec_angle2(a,b))     # is: 90 ok
np.degrees(vec_angle2(b,a))     # is: -90 ok

#  3. Quadrant
a = np.array([-1,-1])
b = np.array([1,-1])

np.degrees(vec_angle2(a,b))     # is:  90 ok
np.degrees(vec_angle2(b,a))     # is: -90 ok

#  4. Quadrant
a = np.array([1,-1])
b = np.array([1,1])

np.degrees(vec_angle2(a,b))     # is:  90 ok
np.degrees(vec_angle2(b,a))     # is: -90 ok



## Look at the parameters of an Agent
n1 = CatchBot()
print("Tau:\n", n1.Tau,
      "\n \n weights:\n", n1.W,
      "\n \n Motor weights:\n", n1.WM,
      "\n \n Vision weights:\n", n1.WV,
      "\n \n Biases:\n", n1.Theta)



## Fitness proportionate selection

e3 = Evolution(simlength=10)
population = e3.pop_list[:,0]
fitness = e3.pop_list[:,1]
fitness = 1-normalize(fitness)

population = np.arange(1,11)
fitness = np.array([98,76,67,53,51,41,32,22,11,9])


total_fitness = sum(fitness)
relative_fitness = [f/total_fitness for f in fitness]

probs = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]

new_population = []
for n in range(len(fitness)):
    r = np.random.random()   # continous uniform distribution
    print(r)
    for (i, individual) in enumerate(population):
        if r <= probs[i]:
            new_population.append(individual)
            break

print(population)
print(new_population)


### Testing Mutation
# from Evolution import Evolution doesnt work: imports everything, instead of just the class

t1 = Evolution(simlength=10)
gens = t1.gen_code()
mutation_var = .001
fts = True

new_population = copy.copy(t1.pop_list)
old_population = copy.copy(t1.pop_list)

AGTC = sum(gens.values()) - gens["U"]   # sum of all gen-sizes, except Tau
U    = gens["U"]                        # == t1.agent.Tau.size

mu, sigma = 0, np.sqrt(mutation_var)    # mean and standard deviation

for i in range(1-fts, new_population.shape[0]):  # if fts = False => range(1,size), else => range(0,size)

    mutation_AGTC = np.random.normal(mu, sigma, AGTC)
    mutation_U    = np.random.normal(mu, sigma, U)

    # plt.hist(mutation_AGTC,17)
    # plt.hist(mutation_U)
    # np.mean(mutation_AGTC)
    # np.mean(mutation_U)

    AGTC_mutated =  new_population[i,2:AGTC+2] + mutation_AGTC

    AGTC_mutated[AGTC_mutated > t1.agent.W_RANGE[1]] = t1.agent.W_RANGE[1]       # Replace values beyond the range with max.range
    AGTC_mutated[AGTC_mutated < t1.agent.W_RANGE[0]] = t1.agent.W_RANGE[0]       # ... or min.range (T_RANGE = W.RANGE =[-13, 13])

    new_population[i,2:AGTC+2] = AGTC_mutated


    U_mutated = new_population[i,(AGTC+2):] + mutation_U

    U_mutated[U_mutated > t1.agent.TAU_RANGE[1]] = t1.agent.TAU_RANGE[1]        # Replace values beyond the range with max.range
    U_mutated[U_mutated < t1.agent.TAU_RANGE[0]] = t1.agent.TAU_RANGE[0]        # ... or min.range (TAU_RANGE = [1, 10])

    new_population[i,(AGTC+2):] = U_mutated


    new_population[i,0] = i+1   # reset enumeration
    new_population[i,1] = 0     # reset fitness



print(np.round(old_population[0:4,2:4] - new_population[0:4,2:4],2))
print(np.mean(old_population[:,2:] - new_population[:,2:]))


## Does work

## Github
# ...
print(24+24)  # Commit: "Test"
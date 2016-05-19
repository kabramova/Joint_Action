from Formulas import *
from RollingBot import CatchBot
from Evolution import Evolution


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

# e3 = Evolution(simlength=10)
# e3.run_evolution(Generations=1)
# population = e3.pop_list[:,0]
# fitness = e3.pop_list[:,1]
population = np.array(range(1,11))
fitness = np.sort(np.array([98,26,67,3,51,2,32,82,11,9]))   # more or less equally distributed fitness
# fitness = np.sort(np.array([98,96,87,3,81,2,72,82,91,9]))   # very biased fitness distribution
pop_mat = np.zeros((np.size(population),2))
pop_mat[:,0] = population
pop_mat[:,1] = fitness

fitness = 1-normalize(fitness)
print(fitness)

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
print(np.mean(new_population))


# FPS for particular area in the pop_list:
new_population = np.zeros(pop_mat.shape)
for n in range(3,7):
    r = np.random.random()   # continous uniform distribution

    for (i, individual) in enumerate(population):
        if r <= probs[i]:
            new_population[n] = individual
            break

print(pop_mat)
print(new_population)



### Testing Mutation
# from Evolution import Evolution doesnt work: imports everything, instead of just the class

t1 = Evolution(simlength=10)
gens = t1.gen_code()
mutation_var = .001
fps = True

new_population = copy.copy(t1.pop_list)
old_population = copy.copy(t1.pop_list)

AGTC = sum(gens.values()) - gens["U"]   # sum of all gen-sizes, except Tau
U    = gens["U"]                        # == t1.agent.Tau.size

mu, sigma = 0, np.sqrt(mutation_var)    # mean and standard deviation

for i in range(1-fps, new_population.shape[0]):  # if fps = False => range(1,size), else => range(0,size)

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


## Creating files, depending on arguments:

t3 = Evolution(simlength=10)
Fitness_progress, pos_target = t3.run_evolution(Generations=10, mutation_var=.25,
                                                complex_trials=True, fit_prop_sel=True,
                                                position_agent=[50,50],
                                                angle_to_target= np.pi/2,  distance_to_target = 30)

t3.filename
t3.reimplement_population(Plot=True)
t3.Generation
t3.simlength

t = t3.filename
int(t[t.find('m')+1 : t.find('.')])


# Works!


## Testing acceleration: Comparison between angle and distance.
# Is the increase of beta(visual angle) linear to the increas of b (distance on screen):


# First we see how distance is translated into the angle (beta):
steps = 500
bs = []
b_angles = []
for b in range(1, steps):
    bs.append(b)
    b_angles.append(angle_velo(b=b))

plt.plot(bs,b_angles, 'r')

cms = []
cm_angles = []
for cm in range(1, steps):
    cms.append(cm)
    cm_angles.append(angle_velo2(velocity_cm=cm))

plt.plot(cms,cm_angles, 'yo')


# Then, we see how the angle is translated into distance (b):
steps = 81
betas = []
beta_b = []
for beta in range(1, steps):
    betas.append(beta)
    beta_b.append(angle_velo(beta=beta))

plt.plot(beta_b,betas, 'b')

thetas = []
theta_cm = []
for theta in range(1, steps):
    thetas.append(theta)
    theta_cm.append(angle_velo2(velocity_deg=theta))

plt.plot(theta_cm, thetas, 'go')

plt.close()

# Now we turn the plot
plt.plot(betas, beta_b, 'b')

# and see how a linear increase of distance with respect of the angle would look like:
bs2 = []
for i in range(1,len(betas)):
    b = beta_b[0]*i     # we take the value of 1degree
    bs2.append(b)

plt.plot(bs2, 'r')

## So or so (angle_velo vs. angle_velo2[more correct and approximate linear increase]),
# we rather want a linear acceleration of the tracker


# Iterate through different conditions:
i = 0
for trial_speed in ["slow", "fast"]:
    for auditory_condition in [False, True]:
        for init_target_direction in [-1, 1]:  # left(-1) or right(1)
            i += 1
            print("____",i,"____")
            print(trial_speed)
            print(auditory_condition)
            print(init_target_direction)
print("___________")
print(i,"different combinations")


## Indexing

pop_size = 111

n_p = 2

n_c = pop_size*0.2 if pop_size*0.2 < 10 else 10

n_f = pop_size * 0.3
n_r = pop_size * 0.3

if (pop_size - (n_p + n_c + n_f + n_r)) != 0:
    rest = pop_size - (n_p + n_c + n_f + n_r)
    if rest % 2 > 0:
        n_f += (rest + 1) / 2
        n_r += (rest - 1) / 2
    else:
        n_f += rest / 2
        n_r += rest / 2

print("pop_size:", pop_size)
print("n_p:", n_p)
print("n_c:", n_c)
print("n_f:", n_f)
print("n_r:", n_r)
print("Sum:", np.sum((n_p, n_c, n_f, n_r)))


##  Test PLotting for JA_Simulator:
simlength = 100
condition = True

positions = np.zeros((simlength, 2))
keypress = np.zeros((simlength, 2))
if condition == True:
    sounds = np.zeros((simlength, 2))

if not os.path.exists("Animation"):
    os.makedirs("Animation")

for i in range(simlength):
    positions[i,0] = i* 0.2  # Tracker...
    positions[i,1] = i*-0.2  # ... and target move in opposite directions

    if i%5 == 0:
        keypress[i,0] = -1
    if i%10 == 0:
        keypress[i,1] = 1

    if i%5 == 0 or i%10 == 0 and condition == True:
        sounds[i,:] = keypress[i,:]

    # Plot:
    if i % 10 == 0:  # should do 9 images

    # For matplotlib see: http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
        plt.figure(figsize=(10, 6), dpi=80)

        plt.plot(positions[i,0],0, 'ro', markersize=12, alpha = 0.5) # Tracker
        plt.plot(positions[i,1],0, 'go') # Target

        if keypress[i,0] == -1:
            plt.plot(-10, -4, 'yo', markersize=24,  alpha = 0.5)  # sound left
            plt.plot(-10, -4, 'bs', markersize=16)  # keypress left


        if keypress[i,1] == 1:
            plt.plot(10, -4, 'yo', markersize=24,  alpha = 0.5)  # sound right
            plt.plot( 10, -4, 'bs', markersize=16)  # keypress right


        plt.xlim(-25, 25)
        plt.ylim(-5, 5)

        plt.annotate(xy=[0,4], xytext=[0,4], s="fitness = 12")

        plt.savefig('./Animation/animation{}.png'.format(int(i/10)))

        plt.close()

# To create gif use Terminal Command in for folder:
# convert -delay 0 -loop 0 animation*.png animated.gif
# Source: http://johannesluderschmidt.de/creating-animated-gifs-from-the-command-line-in-mac-os-x-using-convert/2732/
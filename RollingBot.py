from Formulas import *
from CTRNN import CTRNN


class CatchBot(CTRNN):


    def __init__(self, position_agent = [50,10], position_target = [50,90]): # *args = number_of_neurons, timestep = 0.01)

        self.N_sensor = 2
        self.N_motor = 2
        self.position = np.array(position_agent)  # its also reasonable to implement this in class environment
        self.Angle = np.pi/2
        self.Velocity = 0
        super(self.__class__, self).__init__(number_of_neurons = 6, timestep = 0.01) # or *args

        self.WM = randrange(self.W_RANGE, self.N_motor*2, 1)  # Motor-Weights, Neuron 3 & 5 to left and right Motor (We could set outer and inner weights equal, respectivly)
        self.WV = randrange(self.W_RANGE, self.N_sensor*2, 1) # Vision-Weights,Neuron 6,1 & 2 to left and right eye (We could set outer weights equal, and inner weights with different signs)

        self.position_target = np.array(position_target)   # as self.position this might should be in another class

        # print("Position Agent:", self.position)
        # print("Position Target:", self.position_target)


    def compute_motor_neurons(self):
        '''
        Input from Neuron 3 and Neuron 5
        :rtype: object
        :return: activation[left motor, right motor]
        '''


        # round(N/2): = attach Motor down right (clockwise enumeration of network 1=Top)
        N3 = self.Y[round(self.N/2) - 1]                # Neuron 3 (if N=6)
        # N - round(N/2) + 2 : = attach Motor down left
        N5 = self.Y[self.N - round(self.N/2) + 2 - 1]   # Neuron 5 (if N=6)

        outer_weights_MR = self.WM[1]   # [N3_MR = 1]
        inner_weights_MR = self.WM[3]   # [N5_MR = 3]

        outer_weights_ML = self.WM[0]   # [N5_ML = 0]
        inner_weights_ML = self.WM[2]   # [N3_ML = 2]

        activation_left  = np.sum([N3*inner_weights_ML, N5*outer_weights_ML])   # or with sigmoid()? If, then leave MaxThrust ?!
        activation_right = np.sum([N3*outer_weights_MR, N5*inner_weights_MR])

        output = np.array([activation_left, activation_right])

        return output


    def visual_input(self):
        '''
        This computes the angle between the target and each eye.
        The right eye diverge from the agent's position by +.02
        The left eye diverge from the agent's position by -.02

        sets I (Input for the Neurons 6, 1 and 2
        '''

        temp_angle = np.array([np.cos(self.Angle), np.sin(self.Angle)])
        rotangle = rotate90(temp_angle, clockwise=True)                         # rotation would work also with: self.Angle - np.pi/2

        eyeL = self.position + (-.2) * rotangle
        eyeR = self.position + (+.2) * rotangle

        Eyes = np.array((eyeL, eyeR))

        angle_target_eyes = np.zeros(np.shape(Eyes)[0])

        # TODO: 1) Angle has no sign yet. 2) We can include the distance to object (the farer away the smaller it is on the retina)
        # TODO: -> 1) vec_angle2(v1,v2) takes sign into account (take care with the order of vectors (v1,v2))
        for i,eye in enumerate(Eyes):

            wT = self.position_target - eye                                 # vector between Target and agent's eye

            angle_target_eyes[i] = vec_angle(temp_angle, wT)                # or with np.degrees() => bigger effect

        # angle_target_eyes as Input for Neurons (weights the same so far, respectivly):
        #   If target is out of the field of vision (>90degrees) then no Input

        # print(angle_target_eyes)

        for j,k in enumerate(angle_target_eyes):
            if k > np.pi/2:    # check whether vec_angle is in degrees (>90 or >np.degrees(np.pi/2)) or not (> np.pi/2)
                angle_target_eyes[j] = 0

        self.I[self.N-1] = self.WV[2] * angle_target_eyes[0]   # to left Neuron 6
        self.I[0] = np.sum( ((self.WV[0] * angle_target_eyes[0]), (self.WV[1] * angle_target_eyes[1])) )  # (middle Neuron 1, we could subtract to let the net calculate the difference between both incomes at this node)
        self.I[1] = self.WV[3] * angle_target_eyes[1]          # to right Neuron 2

        return angle_target_eyes


    def movement(self, number_of_movements = 1):

        '''
        ## Agmon & Beer (2013):
        Torque: = (Output(Motor_right) - Output(Motor_left)) * MaxAngle , MaxAngle  = Pi/12
        Thrust: = (Output(Motor_right) + Output(Motor_left)) * MaxThrust, MaxThrust = 0.008
        angle(t) = angle(t-1) + StepSize*Torque
        Velocity(t) = Velocity(t-1)*0.9 + Stepsize*Thrust (Friction coefficient = 0.9 (slows agents if no motor-activity))

        new_position = old_position + [cos(aplha), sin(alpha)]' * v * h [CHECK]

        After a movement the next_state will be computed as well as the input (I) of the visual system at the new position
        '''

        for _ in range(number_of_movements):

            MaxAngle  = np.pi/12
            MaxThrust = 0.008

            Torque = (self.compute_motor_neurons()[1] - self.compute_motor_neurons()[0]) * MaxAngle
            Thrust = (self.compute_motor_neurons()[1] + self.compute_motor_neurons()[0]) * MaxThrust

            oldVelo = self.Velocity
            self.Velocity = oldVelo * 0.9 + Thrust * self.h

            oldAngle = self.Angle
            self.Angle = np.mod((oldAngle + Torque * self.h), 2*np.pi) # we normalize with modulus operator: Angle ∈ [0,2π]

            oldPos = self.position
            self.position = oldPos + np.array([np.cos(self.Angle), np.sin(self.Angle)]) * self.Velocity * self.h

            # After a movement, the visual input at the new position and then the next state will be computed (ordered !)
            self.visual_input()  # 1. step
            self.next_state()    # 2. step


    def show_innards(self, rounds=False):
        if not rounds:
            print("Tau:\n", self.Tau,
                  "\n \n weights:\n", self.W,
                  "\n \n Motor weights:\n", self.WM,
                  "\n \n Vision weights:\n", self.WV,
                  "\n \n Biases:\n", self.Theta)
        else:
            print("Tau:\n", np.round(self.Tau,2),
                  "\n \n weights:\n", np.round(self.W,2),
                  "\n \n Motor weights:\n", np.round(self.WM,2),
                  "\n \n Vision weights:\n", np.round(self.WV,2),
                  "\n \n Biases:\n", np.round(self.Theta,2))




'''
class Environment:
    def __init__(self):
        pass

## class Environment (100X100)

# Starting position Bot and goal:
START_xy = np.matrix((np.arange(5,100,10),np.repeat(10,10) )) # x coordinate, row =1 # y coordinate, row =2
Goal_xy = np.matrix(([50],[90]))
'''
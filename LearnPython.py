import numpy as np

# array
ar = np.array([1,2,3])          # array
ar2 = np.array((1,2,3))         # array
ar == ar2
ar[1]
br = np.array([[1,2,3]])
br[0][2]
am = np.array([[1, 2], [3, 4]]) # matrix
print(ar), print(ar2), print(br), print(am)

# whether all/any elements are True/False:
if np.any(ar==3):
    print("at least one element == 3")

if np.all(ar != 0):
    print("none of the elements == 0")


# find min/max in Array
np.min(ar)
np.argmin(ar) # index of min
np.argmax(ar) #

# Connect (concatenate) arrays/matrices
np.concatenate((ar,ar))
np.concatenate((am,am), axis=0)
np.concatenate((am,am), axis=1)

# Transpose
np.transpose(am)
am.T
ar.T  # equivalent !
ar
br.T

## Add column/row to matrix

X = np.zeros((3,2))
Y = np.ones((3,1))
np.c_[X,Y]
np.append(X,Y, axis=1)

## Fill diagonal of matrix
am
np.fill_diagonal(am,99)
am

## Sort Matrix

M = np.reshape(np.array([3,2,5,7,1,2,9,8,4]),(3,3))
M
M[ np.argsort(M[:,1])] # whole matrix get sorted with respect to the second column
# sorting by row just works with transposing the matrix
M
np.transpose(np.transpose(M)[ np.argsort(np.transpose(M)[:,1])])


# Use Matrices for iteration, enumerate
for i in enumerate(ar): print (i)
print("\n")
for i,j in enumerate(ar): print (i,j)
for i,j in enumerate(am):
    print (i,j)
    print (i)
    print (j)

# Stop iteration
for i in [1,2,3]:
    if i != 3:
        print(i)
    else:
        StopIteration

for i in [1,2,3]:
    if i != 3:
        print(i)
    else:
        break

# Random array/matrix
R_RANGE = [-5, 5]
(R_RANGE[1] - R_RANGE[0]) * np.random.sample((2,2)) +  R_RANGE[0]  # (b - a) * random_sample() + a, b>a
np.random.rand(2,3) # [0,1]
np.random.random() # single values
np.random.rand()
np.random.sample()


## Conditional variable setting
a = 0
b = a if a!=0 else 2   # Note "else" is obligatory here.
c = a if a==0 else 1
print("a:", a,
      "b:",b,
      "c:", c)

# Free space
print("\n")


# Analyse Object
import inspect

type(ar)
dir(ar)

# Inputs User
input()

l = input("Enter some numbers:")
l = l.split()
sum(int(i) for  i in l)

count = 0
while True and count != 3:
    l = input("Only numbers: ")
    try:
        print(int(l))
        break
    except ValueError:
        print ("No Number, try {} more times".format(2-count))
        count += 1


def foo(a,b,c):
    d = a+b+c

    count = 0
    while True and count != 3:
        Input = input("Do you want to save the final population ('(y)es'/'(n)o'):")

        if Input in ["y", "Y","yes" ,"Yes", "YES"]:
            # pickle.dump(d, open('d.sim{}.mut{}.Gen{}'.format(a,b,c),      'wb'))
            print("successfully saved")
            break
        elif Input in ["n", "N", "no", "No", "NO"]:
            print("Final population won't be saved in external file")
            break
        else:
            print("Input is not understood.\n"
                  "Type either 'yes' or 'no'.\n"
                  "{} more attempts".format(2-count))
            count += 1
    if count == 3:
        raise ValueError("Function stopped")

    return d

foo(100,.001,200)


# Current working directory
import os
os.getcwd()


## Work with formaters:

# Fill word in
print("That is a {}, isn't it?".format("PLACEHOLDER"))


# Space formater:
print("Space is {:>10}".format("GREAT"))
print("Space is {:>6}".format("GREAT"))  # "Great" has 5 characters, so +1 space
print("Space is {:>5}".format("GREAT"))  # no extra space

print("{:10} Space".format("GREAT"))
print("{:5} Space".format("GREAT"))

# Define size of space
print("{:<{}s} Space".format("GREAT",10))
print("{:{}s} Space".format("GREAT",10))
print("{:{}s} Space".format("GREAT",len("GREAT")+5))

print("Space is{:>{}s}".format("GREAT",10))  # {...s} is not necessary
print("Space is{:>{}}".format("GREAT",10))
print("{:{}s} Space".format("GREAT",10))

# fill space with ., ,,,_, :
print("{:_<10} Space".format("GREAT"))
print("{:_>10} Space".format("GREAT"))
print("Space is {:.>10}".format("GREAT"))

# space left and right
print("{:.^10}".format("SPACE"))
print("{:_^11}".format("SPACE"))
print("{:^11}".format("SPACE"))

# shorten long words
print("This is an {:.4}".format("abbrevation"))
print("This is an {:.{}}".format("abbrevation",4))

print("This is an {:>8.4}".format("abbrevation"))

print("What is the word: {:_<{}.{}}".format("abbrevation",len("abbrevation"), 4))

# Numbers
print("{:d}".format(42))        # d is not necessary
print("{}".format(42))
print("{}".format(42.3456))
print("{:f}".format(42.3456))  # f for float

print("{:12f}".format(42.3))
print("{:12}".format(42.3))
print("{:12d}".format(42))
print("{:12}".format(42))

print("{:0.2f}".format(42.3))
print("{:0.3f}".format(42.3))
print("{:010.3f}".format(42.3))
print("{:010.2f}".format(42.3))
print("{:_>10.2f}".format(42.3))
print("{:_<10.2f}".format(42.3))

print("{:+.2f}".format(42.3))

print("{:2f}".format(42.3))
print("{: 2f}".format(42.3))

print("{:=5}".format(-42))
print("{:=10}".format(-42))


# Placeholder Dicts
Data = {"a": "ABC",
        "b": "123"}
print("Learn the {a} and counting from {b}".format(**Data)) # ** for keyword arguments

Data2 = ("ABC", "123")
print("Learn the {} and counting from {}".format(*Data2))   # * for arguments
print("Learn the {0} and counting from {1}".format(*Data2))
print("Learn the {1} and counting from {0}".format(*Data2))
print("Learn the {0!s} and counting from {1!r}".format(*Data2))   # s string, r represent


# Placeholders with classes

class ABC(object):

    def a(self):
        return "A"

    def b(self):
        return "B"

    type = "c"

print("first {0!s}, then {0!r}".format(ABC().a()))
print("first {0!s}, then {0!r}".format(ABC().b()))
print("first {0!s}, then {1!r}".format(ABC().a(),ABC().b()))
print("first {1!s}, then {0!r}".format(ABC().a(),ABC().b()))
print("first {1!r}, then {0!s}".format(ABC().a(),ABC().b()))
print("C is {p.type}".format(p=ABC()))



# open, read, write Files
f = open("Testtext.txt", "r+") #"r+" = mode, to read and write.
f.read()
f.seek(0)
f.readline()
f.readline()
f.tell()    # returns an integer giving the file object’s current position in the file represented as number of bytes
f.write("\n This is a test")
f.seek(0)   # To change the file object’s position, use f.seek(offset, from_what). The position is computed from adding
            # offset to a reference point; the reference point is selected by the from_what argument. A from_what value
            # of 0 measures from the beginning of the file, 1 uses the current file position, and 2 uses the end of
            # the file as the reference point. from_what can be omitted and defaults to 0, using the beginning of the
            # file as the reference point.
f.tell()
f.read()

f = open("Testtext.txt", "a") # "a" for appending
f.write("This is a test")

# Dealing with text
p = "Das ist ein Text" # String with n Wörter
p = p.split() # turns it into list with n elements
sorted(p)
p.pop(0) # prints first word and deletes it

# argv
from sys import argv
argv

# Conditioning
p = "Das ist ein Text" # String with n Wörter
p = p.split() # turns it into list with n elements

print(p)
for i in range(1,len(p)):
    if len(p) < 3:
        print("kein Satz mehr")
    elif len(p) < 4:
        print ("unvollständiger Satz")
    else:
        print("Und ein langer Satz")

    p.pop(-1)
    print(p)
print("ist ein Wort")




# raise error / exceptions (2 ways)

def a_b(a,b):
    if a>b:
        raise ValueError("a is bigger than b")
    print((a,b))

a_b(1,2)
a_b(2,1)

def b_c(b,c):
    assert b<c, "b is bigger than c"
    print((b,c))

b_c(1,2)
b_c(2,1)


# Classes

class Learn:
    def __init__(self,input):
        self.a = "first letter"
        self.b = "second letter"
        self.c = input

    def a_plus_b(self):
        return print(np.concatenate(([self.a],[self.b])))

    def c_plus_ab(self):                                            # all three lines produce the same
        return print(np.append(self.a_plus_b(), self.c),            # self.method()
                     np.append(Learn.a_plus_b(self), self.c),       # ClassName.method(self)
                     np.append(__class__.a_plus_b(self), self.c))   # __class__.method(self)  (abstact form)




class Learn2(Learn):  # inherit from parent class Learn
    def __init__(self, *args):                     # or input
        super(self.__class__,self).__init__(*args) # or input

    def a_plus_c(self):
        return print(np.concatenate(([self.a],[self.c])))

abc = Learn("THIRD LETTER")
print(abc.a)
print(abc.b)
print(abc.c)
abc.a_plus_b()
abc.c_plus_ab() # Here something doesnt work with the stacked operation
ac = Learn2("3d Letter")
print(ac.a)
print(ac.b)
print(ac.c)
ac.a_plus_b()
ac.a_plus_c()


## Boolean

1 == 1 and  1 == 2
1 == 1 and 1 == int("1")
2 == 1 or 2 == 3
2 == 2 or 2 == 3
2 == 3 or 2 == 2

1 != 2

not 1 != 1

# XOR
bool(1 == 1) ^ bool(1 == 2)
bool(1 == 1) ^ bool(1 != 2)






## Class properties

class Hide:

    def __init__(self, t = "transpartent"):
        self.s = "sichtbar"
        self.t = t
        self.__u = "unsichtbar"

    def getT(self):
        return self.__t

    def getU(self):
        return self.__u

    def setT(self, transformer):
        if not isinstance(transformer, str):
           raise ValueError("It has to be a string")
        self.__t = transformer
        print("set Values")

    t = property(getT, setT, doc = "Blablabla")


H = Hide()
H1 = Hide ("TRANSPARENT")
H.s
H.t
H.__u         # is not reachable like that
H.getU()
H.getT()    # ... but on this way
H.setT(23)
H.setT("durchsichtig")
H.getT()
H.t
H1.t

# better to write as...

class Hide:
    def __init__(self, u = "unsichtbar"):
        self.s = "sichtbar"
        self.__u = u

    def __x(self):
        print("I am a method in Hide, which is just approachable by '._Hide__method' ")

    def _y(self):
        print("I am a method in Hide, which should not be touched, indicated by '_' ")

    @property
    def u(self):
        print("Getting value from its hideout")
        return self.__u

    @u.setter
    def u(self, u):
        if not isinstance(u, str):
           raise ValueError("It has to be a string")
        else:
            print("Setting Value of the hidden object")
            self.__u = u

    @u.deleter
    def u(self):
        del self.__u


H = Hide()
H1 = Hide("TRANSPARENT")
H.s
H.__u         # is not reachable like that
H._Hide__u    # but like that
H.u           # or with the getter
H.u = 23      # doesnt work because no string
H.u = "transformed"
H.u
del H.u       # object deleted
H.u
H.u = "rebirth"
H.u

H1.u
H._y()
H.__x()
H._Hide__x()


# more hidden stuff (functions):
class ABC:
    def __index__(self):
        pass

    def abc(self, arg):
        b = arg
        def intern(b):
            return b+2
        return intern(b)

    def summe_AB(self,A,B):
        return A+B

    def __hidden(self, A):
        return A + 2

    def use_hidden(self,A,B):
        print("Hidden = 2 +",A," = " ,self.__hidden(A))
        print("Solution is:")
        return B + self.__hidden(A)

A = ABC()
A.abc(0)
A.summe_AB(3, A.abc(0))   # ergo: argument can be a method
A.__hidden(2)  # does not work
A.use_hidden(2,3)
A.use_hidden(4,2)


## Class and attributes:

class Get:
    a = 5                   # class-level attribute
    def __init__(self,b):
        self.b = b          # instance-level attribute

    def adds(self):
        return self.a + self.b

get = Get(1)
get.a
get.b

Get.a = 7     # The attribute changed now for all instances...

get2 = Get(2)
get2.a       # ... for new instances,
get.a        # but also for old ones !!
get2.b
get2.adds()
get.adds()


## Methods

def foo(x,y=5):   # y=5 default value
    return print(x+y)

foo(2)
foo(2,6)

# Method with different functions
def meth(x,y, *args):
    for arg in args:
        return arg(x,y)

def summe(x,y):
    return x+y

def minus(x,y):
    return x-y

meth(7,3,summe)
meth(7,3,minus)
meth(7,3,summe) + meth(7,3,minus)

def sum_all(a,b,*args):
    Summe = sum((a,b))
    for arg in args:
        Summe += arg
    return Summe

sum_all(3,4)
sum_all(3,4,3)
sum_all(3,4,3,10)



## Special function
data = ["4", "5", "6"]
sum(int(i) for i in data)
data = [4, 5, 6]
np.sum(i for i in data)




## Store an object externaly.

import pickle
lis = np.array([1,2,3])
pickle.dump(lis, open('lis', 'wb'))
relis = pickle.load(open('lis','rb'))



## Dictionary

dic1 = {"a":1,
        "b":2,
        "c":3}
print(dic1)    # wrong order


from collections import OrderedDict

dic2 = OrderedDict([("a", 1),
                    ("b", 2),
                    ("c", 3)])
print(dic2)    # right order


## How quick is the code:

import datetime

start = datetime.datetime.now()

for i in range(1000):
    for j in range(1000):
        if np.power(i/4,2)==np.sqrt(j*4):
            print(i,j)

finish = datetime.datetime.now()

print(finish-start)

# line_profiler...

# Packages:
# For Computer Vision:
# import SimpleCV

# for Neural Networks
# FANN, neurolab
# https://wiki.python.org/moin/PythonForArtificialIntelligence


## 3D plot with matplotlib
# ...
# fig = plt.figure()
# ax = Axes3D(fig)
# x,y = np.mgrid[:z.shape[0], :z.shape[1]]

# ax.plot_surface(x, y, z, rstride = 1, cstride=1, cmap = cm.jet)

# plt.imshow()   (for black& white pictures depending on Matrix (1,0)



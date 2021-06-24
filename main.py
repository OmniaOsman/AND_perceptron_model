import numpy as np


# data set of AND logic gate
# input  | output
#  0 0   |   0
#  1 0   |   0
#  0 1   |   0
#  1 1   |   1

# Binary Step Activation Function
def binaryStep(v):
    if v >= 0:
        return 1
    return 0


def perceptronModel(x, w, b):
    v = np.dot(w, x) + b
    return binaryStep(v)


# initial weight [1, 1] and bias -1.5
def AND_logicFunction(x):
    w = np.array([1, 1])
    bAND = -1.5
    return perceptronModel(x, w, bAND)


# Note: if bias = -0.5 it will be OR logic gate

# inputs
input1 = np.array([0, 0])
input2 = np.array([1, 0])
input3 = np.array([0, 1])
input4 = np.array([1, 1])


print('AND(0, 1) = ', AND_logicFunction(input1))
print('AND(1, 0) = ', AND_logicFunction(input2))
print('AND(0, 1) = ', AND_logicFunction(input3))
print('AND(1, 1) = ', AND_logicFunction(input4))

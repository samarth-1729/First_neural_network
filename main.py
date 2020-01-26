import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def diff_sigmoid(x):
    return np.exp(-x)/(1+np.exp(-x))

# Training Inputs

inputs = np.array([[1,0,0,1,1], [1,1,0,1,1], [1,0,1,0,1], [1,1,1,0,1], [1,1,1,1,1]])

# Training Outputs

correct_output = np.array([[1,1,1,0,0]]).T # T is for transpose

# Random Synpatic Weights
np.random.seed(1)
synaptic_weight = 2*np.random.random((5,1)) - 1

print("Random Weight: \n %s" % synaptic_weight)

# Back Propogation

for i in range(3000):
    outputs = sigmoid(np.dot(inputs, synaptic_weight))
    error = correct_output - outputs
    adjust = error*diff_sigmoid(outputs)
    synaptic_weight = synaptic_weight + np.dot(inputs.T, adjust)

print("Outputs After Training : \n %s" % outputs)

mystery_input = np.array([[1,0,0,1,1]])

predition = sigmoid(np.dot(mystery_input, synaptic_weight))

print("predition after training: \n %s" % predition)

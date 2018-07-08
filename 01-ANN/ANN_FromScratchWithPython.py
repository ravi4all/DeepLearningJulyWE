import numpy as np

X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
y=np.array([[1],[1],[0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return x * (1 - x)

epoch = 10000
learning_rate = 0.1
input_neurons = 4
hiddenlayer_neurons = 3
output_neurons = 1

# weight for hidden layer
wh = np.random.uniform(size=(input_neurons, hiddenlayer_neurons))
# bias for hidden layer
bh = np.random.uniform(size=(1,hiddenlayer_neurons))
# weight for output layer
wout = np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
# bias for output layer
bout = np.random.uniform(size=(1,output_neurons))


for i in range(epoch):
    # feedforward
    hiddenlayer_input_1 = np.dot(X,wh)
    hiddenlayer_input = hiddenlayer_input_1 + bh
    hiddenlayer_activation = sigmoid(hiddenlayer_input)
    
    outputlayer_input = np.dot(hiddenlayer_activation,wout)
    outputlayer = outputlayer_input + bout
    output = sigmoid(outputlayer)
    
    # Backpropagation
    E = y - output
    slope_at_hidden = derivative_sigmoid(hiddenlayer_activation)
    slope_at_output = derivative_sigmoid(output)
    delta_output = E * slope_at_output
    
    Eh = delta_output.dot(wout.T)
    delta_hidden = Eh * slope_at_hidden
    
    wout += np.dot(hiddenlayer_activation.T, delta_output) * learning_rate
    bout += np.sum(delta_output, keepdims=True) * learning_rate
    
    wh += np.dot(X.T, delta_hidden) * learning_rate
    bh += np.sum(delta_hidden, keepdims=True) * learning_rate





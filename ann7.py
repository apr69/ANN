import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

epochs = 10000
lr = 0.1

input_neurons = 2
hidden_neurons = 5
output_neurons = 1

w_ih = np.random.uniform(size=(input_neurons, hidden_neurons))
w_ho = np.random.uniform(size=(hidden_neurons, output_neurons))
b_h = np.random.uniform(size=(1, hidden_neurons))
b_o = np.random.uniform(size=(1, output_neurons))

for _ in range(epochs):
    hidden_layer_input = np.dot(X, w_ih) + b_h
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, w_ho) + b_o
    output_layer_output = sigmoid(output_layer_input)
    
    error = y - output_layer_output
    d_output = error * sigmoid_derivative(output_layer_output)
    
    error_hidden = d_output.dot(w_ho.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)
    
    w_ho += hidden_layer_output.T.dot(d_output) * lr
    w_ih += X.T.dot(d_hidden) * lr
    
    b_o += np.sum(d_output, axis=0) * lr
    b_h += np.sum(d_hidden, axis=0) * lr

output_layer_output = np.round(output_layer_output).astype(int)

print("Predicted Output:")
print(output_layer_output)

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_propagation(x, wih, bih, who, bho):
    hidden_output = sigmoid(np.dot(x, wih) + bih)
    output = sigmoid(np.dot(hidden_output, who) + bho)
    return hidden_output, output

def backward_propagation(x, y, hidden_output, output, wih, who, bih, bho, lr):
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)
    
    hidden_error = np.dot(output_delta, who.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
    
    who += np.dot(hidden_output.T, output_delta) * lr
    bho += np.sum(output_delta, axis=0, keepdims=True) * lr
    wih += np.dot(x.T, hidden_delta) * lr
    bih += np.sum(hidden_delta, axis=0, keepdims=True) * lr

input_size = 2
hidden_size = 3
output_size = 1

wih = np.random.randn(input_size, hidden_size)
bih = np.random.randn(1, hidden_size)
who = np.random.randn(hidden_size, output_size)
bho = np.random.randn(1, output_size)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) #And Gate
targets = np.array([[0], [0], [0], [1]])

epochs = 1000
lr = 0.1

for epoch in range(epochs):
    hidden_output, output = forward_propagation(inputs, wih, bih, who, bho)
    
    backward_propagation(inputs, targets, hidden_output, output, wih, who, bih, bho, lr)
    
    loss = np.mean(np.square(targets - output))


test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
hidden_output, predictions = forward_propagation(test_input, wih, bih, who, bho)
print("Predictions after training:")
print(np.round(predictions).astype(int))




difference = targets - predictions
accuracy = 0
for i in range(len(difference)):
    accuracy += difference[i][0]

accuracy = (1 + accuracy/len(difference))*100
print("Average Accuracy of predictions: ",accuracy)

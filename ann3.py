import numpy as np

def check_activation(output):
    if output >= 0:
        return 1
    else:
        return 0

def testing_prediction(inputs,weights,bias):
    predictions = []

    for i in range(len(inputs)):
        weighted_sum = np.dot(inputs[i],weights) + bias
        print("Weighted Sum : ",weighted_sum)
        predictions.append(check_activation(weighted_sum))
    return predictions

def training_predictions(inputs):
    epochs = 1000
    bias = 1
    length_inputs = len(inputs)
    binary_input = len(inputs[0])
    expected_output = np.array([1,0,1,0,1,0,1,0,1,0])
    weights = np.random.rand(binary_input)
    weighted_sum = 0
    learning_rate = 0.1

    for epoch in range(epochs):
        for j in range(length_inputs):
            weighted_sum = np.dot(inputs[j],weights) + bias
            predict = check_activation(weighted_sum)
            error = expected_output[j] - predict
            weights = weights + learning_rate*(error)*inputs[j]
            bias = bias + learning_rate*error

    return weights,bias



inputs = np.array([[0,0,1,1,0,0,0,0],
                   [0,0,1,1,0,0,0,1],
                   [0,0,1,1,0,0,1,0],
                   [0,0,1,1,0,0,1,1],
                   [0,0,1,1,0,1,0,0],
                   [0,0,1,1,0,1,0,1],
                   [0,0,1,1,0,1,1,0],
                   [0,0,1,1,0,1,1,1],
                   [0,0,1,1,1,0,0,0],
                   [0,0,1,1,1,0,0,1]])



weights,bias = training_predictions(inputs)
print(weights)
print(bias)


print(testing_prediction(inputs,weights,bias))

import numpy as np
import matplotlib.pyplot as plt

def step(x):
    return np.where(x>0, 1, 0)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)

def leaky_relu(x):
    return np.where(x>0 , x , 0.01*x)

def softmax(x):
    return np.exp(x) / np.sum (np.exp(x), axis = 0 )


x = np.linspace(-5,5,100)

step_y = step(x)
sigmoid_y = sigmoid(x)
tanh_y = tanh(x)
relu_y = relu(x)
leaky_relu_y = leaky_relu(x)
softmax_y = softmax(x)

plt.figure(figsize=(15,10))
plt.style.use('_mpl-gallery')

plt.subplot(2,3,1)
plt.plot(x,step_y,label = 'Step Function',color = 'red')
plt.title('Step Activation Function')
plt.legend(borderaxespad=1, borderpad=1)

plt.subplot(2,3,2)
plt.plot(x,sigmoid_y,label = 'Sigmoid Function',color = 'green')
plt.title('Sigmoid Activation Function')
plt.legend(borderaxespad=1, borderpad=1)

plt.subplot(2,3,3)
plt.plot(x,tanh_y,label = 'TanH Function',color = 'orange')
plt.title('Hyperbolic Tangent Activation Function')
plt.legend(borderaxespad=1, borderpad=1)

plt.subplot(2,3,4)
plt.plot(x,relu_y,label = 'ReLu Function',color = 'blue')
plt.title('Rectified Linear Unit Activation Function')
plt.legend(borderaxespad=1, borderpad=1)

plt.subplot(2,3,5)
plt.plot(x,leaky_relu_y,label = 'Leaky ReLu Function',color = 'cyan')
plt.title('Leaky Rectified Linear Unit Activation Function')
plt.legend(borderaxespad=1, borderpad=1)

plt.subplot(2,3,6)
plt.plot(x,softmax_y,label = 'Softmax Function',color = 'black')
plt.title('SoftMax Activation Function')
plt.legend(borderaxespad=1, borderpad=1)

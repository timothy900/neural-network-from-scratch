'''  
Created on May 16, 2020
Trying to implement a neural network from scratch.

May 23 version 
- added cost function
'''

# To do:
# 	Backpropagation(Stochastic GD?)

import numpy as np
from numpy import exp
import random


# np.random.seed(0)
random.seed(0)

class Neuron:
	def __init__(self, n_inp):
		# random list of numbers, which represents the weights this neuron and
		# all the neurons in the previous layer
		self.weights = np.array([random.randint(-100, 100)/100 for i in range(n_inp)])
		# this neuron's bias
		self.bias = random.randint(-100, 100)/100


class Layer:
	def __init__(self, n_inp, n_neurons):
		self.neurons = [Neuron(n_inp) for i in range(n_neurons)]
		# matrix dimensions:
		# rows = number of neurons in layer
		# columns = number of weights between neuron and all neurons in previous layer
		# 
		# each row is the set of weights of a neuron of this layer
		self.weight_matrix = np.array([neuron.weights for neuron in self.neurons])
		self.biases = np.array([neuron.bias for neuron in self.neurons])


class NeuralNetwork:
	def __init__(self, *args):
		n_neurons = args
		self.n_neurons = n_neurons
		self.layers = []
		self.inp_layer = args[0]
		# create the corresponding layers
		for i in range(len(n_neurons)-1):
			n_in = n_neurons[i]
			n_out = n_neurons[i+1]
			self.layers.append(Layer(n_in, n_out))
		# activation function: sigmoid
		self.sigmoid = lambda arr: 1/(1+exp(-arr))

	def ff_layer(self, inp, layer):
		# multiply weight matrix of first layer with input
		layer_output = np.dot(layer.weight_matrix, inp) + layer.biases
		return self.sigmoid(layer_output)

	def feedforward(self, inp):
		for i, layer in enumerate(self.layers):
			# for the first layer
			if i == 0:
				output = self.ff_layer(inp, layer)
			else:
				output = self.ff_layer(output, layer)
		return output


	def cost_one(self, guessed_output, actual_output):
		# returns the cost of one training example
		try:
			# find the mean squared error
			return sum([(guessed_output[i]-actual_output[i])**2 \
						for i in range(len(guessed_output))])
		except IndexError: 
			return None

	def backprop(self, x, y):
		# return the avg gradient of the cost function with respect to the weights and biases
		pass

	def train(self, batch_siz, x, y):
		pass
		# trains the network on the training data, x is the input, y is the actual output/answer
		# divide examples into batches of size batch_siz
		# for each batch:
			# total change = 0
			# for each item in batch:
				# take the cost of the input when fedforward through the network
				# compute the negative gradient using backpropagation
				# compute the change(to the network) required by this training example
				# add the change required to the toal change
			# update the network based on the total change



# create a 3-layer network with
# 3 neurons on the first/input layer, 
# 4 on the second/hidden layer, 
# 2 on the third/output layer
network = NeuralNetwork(3, 4, 2)
#network = NeuralNetwork(1,1,1,1)

print("Output of second layer after feeding the input [.5, .9, -.6]")
second_layer_out = network.ff_layer([.5, .9, -.6], network.layers[0])
print(second_layer_out, "\n")

print("Output of third layer after feeding the output of the second layer")
print(network.ff_layer(second_layer_out, network.layers[1]), "\n")

print("Output of neural network after feeding the input [.5, .9, -.6]")
print(network.feedforward([.5, .9, -.6]))


# # print the set of weights and biases of the network
# params = 0
# for i, layer in enumerate(network.layers):
# 	print(f"{i}th layer:")
# 	print("weights\n", layer.weight_matrix)
# 	print("biases\n", layer.biases, "\n")
# 	params += np.shape(layer.weight_matrix)[0] * np.shape(layer.weight_matrix)[1]
# 	params += np.shape(layer.biases)[0]
# print(params)

# print(network.feedforward([]))

# reset seed for files that import this(if that's how it works)
random.seed(0)

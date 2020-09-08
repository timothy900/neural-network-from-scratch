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

	def backprop_layer(self, x, y):
		pass

	def backprop(self, x, y):
		# params:
		# x = single training example input
		# y = correct output of that example

		# return the negative gradient of the cost function with respect to 
		# the weights and bias of one example
		z1 = self.ff_layer(x, nn.layers[0])
		a1 = self.sigmoid(z1)

		z2 = self.feedforward(x)
		a2 = self.sigmoid(z2)

		# dzdw1, dzdb1 = z1, 1
		# dadz1 = a1 * (1 - a1)
		# dcda1 = 2 * (a1 - )

		# dcdw1 = 
		# dcdb1 = 


		# DERIVATIVE OF COST WITH RESPECT TO WEIGHTS = 
		# derivative of weighted sum with respect to weights *
		dzdw, dzdb = z2, 1
		# derivative of activation with respect to weighted sum *
		dadz = a2 * (1 - a2)
		# derivative of cost with respect to activation 
		dcda = 2 * (a2 - y)

		dcdw2 = dzdw * dadz * dcda
		dcdb2 = dzdb * dadz * dcda
		return dcdw2, dcdb2#, [dcdw1, dcdb1]


	def train(self, batch_size, x, y):
		pass
		# trains the network on the training data, x is the input, y is the actual output/answer
		# divide examples into batches of size batch_size
		# for each batch:
			# total change = 0
			# for each item in batch:
				# take the cost of the input when fedforward through the network
				# compute the negative gradient using backpropagation
				# compute the change(to the network) required by this training example
				# add the change required to the toal change
			# update the network based on the total change


nn = NeuralNetwork(1,1,1)

print(nn.feedforward([.5]))
print([(i.neurons[0].weights, i.neurons[0].bias) for i in nn.layers], "\n")


for i in range(10):
	change = nn.backprop(1,1)
	# print(change)
	nn.layers[-1].neurons[0].weights[0] += change[0][0][0]
	nn.layers[-1].neurons[0].bias += change[1][0][0]
# print(nn.feedforward([1]))


print(nn.feedforward([.5]))
print([(i.neurons[0].weights, i.neurons[0].bias) for i in nn.layers], "\n")

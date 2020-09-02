'''  5.18.20 backup
Created on May 16, 2020
Trying to implement a neural network from scratch.
'''

# To do:
#	NeuralNetwork() class
# 	Visualize network
# 	Feedforward through Network
# 	Activation functions
# 	Backpropagation(Stochastic GD?)
#	Use numpy(?)


import random
random.seed(0)


class Neuron:
	def __init__(self, n_inp):
		self.weights = [random.randint(-100, 100)/100 for i in range(n_inp)]
		self.bias = random.randint(-100, 100)/100


class Layer:
	def __init__(self, n_inp, n_neurons):
		self.neurons = [Neuron(n_inp) for i in range(n_neurons)]


class NeuralNetwork:
	def __init__(self, *args):
		n_neurons = args
		self.layers = []
		# create the corresponding layers
		for i in range(len(n_neurons)-1):
			n_in = n_neurons[i]
			n_out = n_neurons[i+1]
			self.layers.append(Layer(n_in, n_out))

	def ff_layer(self, inp, layer):
		# list of outputs of each neuron
		layer_output = []

		# pass input through a layer
		# for each neuron in the layer
		for neuron in layer.neurons:

			# multiply each input by each weight
			for i in range(len(inp)):
				neuron_output = inp[i] * neuron.weights[i]
			# add the bias
			neuron_output += neuron.bias
			# add the output to the layer output, after rounding off
			layer_output.append(round(neuron_output, 3))

		return layer_output


	def feedforward(self, inp):
		# ff_layer() on every layer
		output1 = self.ff_layer(inp, self.layers[0])
		output2 = self.ff_layer(output1, self.layers[1])
		return output2

		# for i in range(len(self.layers)):
		# 	layer = self.layers[i]
		# 	# for the first layer
		# 	if i == 0:
		# 		output = self.ff_layer(inp, layer)
		# 	else:
		# 		output = self.ff_layer(output, layer)
		# return output


def feedforward_layer(inp, layer):
	# list of outputs of each neuron
	layer_output = []

	# pass input through a layer
	# for each neuron in the layer
	for neuron in layer.neurons:

		# multiply each input by each weight
		for i in range(len(inp)):
			neuron_output = inp[i] * neuron.weights[i]
		# add the bias
		neuron_output += neuron.bias
		# add the output to the layer output, after rounding off
		layer_output.append(round(neuron_output, 3))

	return layer_output


# 3-layer network:
# 3 input neurons, 
# 4 hidden neurons, 
# 2 output neurons
# nn = Layer(3, 4)
# nn2 = Layer(4, 2)

# print(nn.neurons[0].weights[0])
# print(nn.neurons[0].bias, "\n")
# print(nn2.neurons[0].weights[0])
# print(nn2.neurons[0].bias, "\n")

# print(feedforward_layer([1, 1.2, .8], nn))

# print( feedforward_layer( feedforward_layer([1, 1.2, .8], nn), nn2 ) )

network = NeuralNetwork(3, 4, 2)

print(feedforward_layer([.5, .9, .6, -.4], network.layers[1]))

print(network.feedforward([.5, .9, .6]))

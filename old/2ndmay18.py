'''  
Created on May 16, 2020
Trying to implement a neural network from scratch.

May 18 backup 2 
- number of layers and neurons in the network are adjustable
- can now feedforward through the network
'''

# To do:
#	NeuralNetwork() class
# 	Visualize network
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
		for i in range(len(self.layers)):
			layer = self.layers[i]
			# for the first layer
			if i == 0:
				output = self.ff_layer(inp, layer)
			else:
				output = self.ff_layer(output, layer)
		return output


# create a 3-layer network with
# 3 neurons on the first/input layer, 
# 4 on the second/hidden layer, 
# 2 on the third/output layer
network = NeuralNetwork(3, 4, 2)


print("Output of second layer after feeding the input [.5, .9, -.6]")
second_layer_out = network.ff_layer([.5, .9, -.6], network.layers[0])
print(second_layer_out, "\n")

print("Output of third layer after feeding the output of the second layer")
print(network.ff_layer(second_layer_out, network.layers[1]), "\n")

print("Output of neural network after feeding the input [.5, .9, -.6]")
print(network.feedforward([.5, .9, -.6]))

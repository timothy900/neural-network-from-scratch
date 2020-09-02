'''
May 19, 2020
- implemented drawNetwork() function

To do:
- clean the code: add variables, make it more readable, fix comments, etc.
- display the weights and biases of the neurons

'''

# Note: I've decided to not put too much time on the style

import neuralnetnumpy
import pygame

pygame.init()
pygame.display.set_caption('neural network visualization')
# screen settings
size = width, height = 800, 600
bg_color = 0, 0, 0
white = (220, 220, 220)
screen = pygame.display.set_mode(size)


neurons_positions = []
def drawNeuron(radius, xy, network):
	# Parameters: 
	# radius of neurons, 
	# xy coordinates(list),
	# neural network object
	x, y = xy[0], xy[1]
	pygame.draw.circle(screen, white, \
	(x, y), radius, 4)
	# prevent function from adding to list every frame
	if len(neurons_positions) < sum(network.n_neurons):
		neurons_positions.append([x, y])

def drawLayer(n_neurons, gap, xy, network):
	# Parameters: 
	# number of neurons, 
	# gap between neurons, 
	# xy of top-middle of layer
	# neural network object
	x, y = xy[0], xy[1]
	for i in range(n_neurons):
		neuron_y = y + (gap+30+20)*i
		drawNeuron(30, [x, neuron_y], network)

def drawLines(network, positions):
	# identify which layers the neurons belong to
	neurons = network.n_neurons
	# neurons is a list of the number of neurons per layer
	# pos_per_layers[n] is the list of positions of neurons on the nth layer
	pos_per_layers = []
	iteration = 0
	for l in neurons:
		layer = []
		for i in range(l):
			layer.append(positions[iteration])
			iteration += 1
		pos_per_layers.append(layer)

	# for each layer except last
	for j, layer in enumerate(pos_per_layers[:-1]):
		next_layer = pos_per_layers[j+1]
		# for each neuron in layer
		for n in range(len(layer)):
			nx, ny = layer[n][0], layer[n][1]
			# for each neuron on the next layer
			for n2 in range(len(next_layer)):
				n2x, n2y = next_layer[n2][0], next_layer[n2][1]
				# draw a line from neuron on current layer to neuron on next layer
				pygame.draw.line(screen, white, \
				(nx, ny), (n2x, n2y), 2)

def drawNetwork(network, radius, neuron_gap, layer_gap):
	# draw input layer
	drawLayer(network.inp_layer, neuron_gap, [100, 100], network)
	# draw the rest of the layers
	for i, layer in enumerate(network.layers):
		drawLayer(len(layer.neurons), neuron_gap, [100+200*(i+1), 100], network)
	# draw the lines connecting the layers
	drawLines(nn, neurons_positions)


nn = neuralnetnumpy.NeuralNetwork(5, 4, 3, 2)

# drawNetwork(nn, 30, 30, 400)
# print(neurons_positions)
# drawLines(nn, neurons_positions)


def main():
	clock = pygame.time.Clock()
	run = True
	while run:
		clock.tick(5)

		for event in pygame.event.get():
			if event.type == pygame.QUIT: 
				run = False

		screen.fill(bg_color)

		drawNetwork(nn, 30, 30, 400)

		pygame.display.flip()

	pygame.quit()
	quit()


main()

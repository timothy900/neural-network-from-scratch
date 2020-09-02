# re-did this code -may 19

import neuralnet
import pygame

pygame.init()

# screen settings
size = width, height = 800, 600
bg_color = 0, 0, 0
white = (220, 220, 220)

screen = pygame.display.set_mode(size)


# drawing settings
# line thickness
thickness = 1
# radius of neuron circle
r = 30
# gap between neurons in layer
neuron_gap = 50


def drawNeuron(radius, xy):
	# Parameters: 
	# radius of neurons, 
	# xy coordinates(list)
	x, y = xy[0], xy[1]
	pygame.draw.circle(screen, white, \
	(x, y), radius, thickness)

def drawLayer(n_neurons, gap, xy):
	# Parameters: 
	# number of neurons, 
	# gap between neurons, 
	# xy of top-middle of layer
	x, y = xy[0], xy[1]
	for i in range(n_neurons):
		neuron_y = y + (gap+r)*i
		drawNeuron(r, [x, neuron_y])


def drawNetwork(nn):
	pass
	

# TEST
net = neuralnet.NeuralNetwork(3, 4, 2)
print("\n", net.layers[0].neurons[0].weights[0])
print("\n", net.layers[0].neurons[0].bias)
drawNetwork(net)
#


def main():
	clock = pygame.time.Clock()
	run = True
	while run:
		clock.tick(60)

		for event in pygame.event.get():
			if event.type == pygame.QUIT: 
				run = False
			if event.type == pygame.MOUSEBUTTONUP:
				pass

		screen.fill(bg_color)
	
		# TEST
		margin = 70
		drawLayer(4, neuron_gap, [int(width/2 - 150), 40+40+margin])
		drawLayer(5, neuron_gap, [int(width/2), 40+margin])
		drawLayer(2, neuron_gap, [int(width/2 + 150), 40+120+margin])
		#

		pygame.display.flip()

	pygame.quit()
	quit()


main()

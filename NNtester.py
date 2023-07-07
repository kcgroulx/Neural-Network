import NeuralNetwork as NN
import math

network = NN.Network([2,3,2])

network.layers[0].weights[0][0] = 1
network.layers[0].weights[0][1] = 2
network.layers[0].weights[1][0] = 3
network.layers[0].weights[1][1] = 4

network.layers[0].bias[0] = 5
network.layers[0].bias[1] = 6

#network.PrintNetwork()

print( network.CalculateOutputs([0.5, 0.1]))
dataset = []
dataset.append(NN.Datapoint([1, 0],[0.5, 0.1]))

print(network.AverageCost(dataset))

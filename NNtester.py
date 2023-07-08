import NeuralNetwork as NN
import math

network = NN.Network([2,3,2])
network.layers[0].weights[0][0] = 1
network.layers[0].weights[0][1] = 1
network.layers[0].weights[1][0] = 1
network.layers[0].weights[1][1] = 1
network.layers[0].bias[0] = 1
network.layers[0].bias[1] = 1
"""
network.layers[1].weights[0][0] = 5
network.layers[1].weights[0][1] = 6
network.layers[1].weights[1][0] = 7
network.layers[1].weights[1][1] = 8


network.layers[1].bias[0] = 11
network.layers[1].bias[1] = 12
"""

datapoints = []
datapoints.append(NN.Datapoint([1,0],[0.1,0.1]))

for i in range(100):
    network.Learn(datapoints, 0.3)

print( network.CalculateOutputs([0.1, 0.1]) )

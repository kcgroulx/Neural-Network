import NeuralNetwork as NN
import math

network = NN.Network([2,3,2])
network.layers[0].weights[0][0] = 1
network.layers[0].weights[0][1] = 1
network.layers[0].weights[1][0] = 1
network.layers[0].weights[1][1] = 1
network.layers[0].bias[0] = 1
network.layers[0].bias[1] = 1


datapoints = []
datapoints.append(NN.Datapoint([1,0],[0.1,0.1]))

print( network.CalculateOutputs([0.1, 0.1]) )

for i in range(10000):
    network.Learn(datapoints, 0.3)

print( network.CalculateOutputs([0.1, 0.1]) )

import random
import math

def Activation(weightedOutput:float):
    if(weightedOutput > 50.0):
        return 1.0
    elif(weightedOutput < -100):
        return 0.0
    else:
        return (1.0 / (1.0 + math.exp(-weightedOutput)))

def nodeCost(outputActivation:float, expectedOutput:float):
       error = outputActivation - expectedOutput
       return (error * error)

class Datapoint:
    def __init__(self, expectedOutput:list[float], inputs:list[float]):
        self.expectedOutput = expectedOutput
        self.inputs = inputs

class Layer:
    def __init__(self , numInputs:int, numOutputs:int):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        #weights[In][Out]
        self.weights = [[random.uniform(-1.0, 1.0) for _ in range(numOutputs)] for _ in range(numInputs)]
        self.bias = [random.uniform(-1.0, 1.0) for _ in range(numOutputs)]
        self.weightCostGradient = [[1.0 for _ in range(numOutputs)] for _ in range(numInputs)]
        self.biasCostGradient = [1.0 for _ in range(numOutputs)]
    
    def PrintLayer(self):
        print("Weights", self.weights)
        print("Biases", self.bias)
        #print("Weight Gradient", self.weightCostGradient)
        #print("Bias Gradient", self.biasCostGradient)

    def CalculateOutputs(self, inputs:list[float]):
        activations = []
        for nodeOut in range(self.numOutputs):
            weightedOutput = self.bias[nodeOut]
            for nodeIn in range(self.numInputs):
                weightedOutput += inputs[nodeIn] * self.weights[nodeIn][nodeOut]
            activations.append(Activation(weightedOutput))
        return activations
    
    def ApplyGradient(self, learnRate:float):
        for nodeOut in range(self.numOutputs):
            self.bias[nodeOut] -= self.biasCostGradient[nodeOut] * learnRate
            for nodeIn in range(self.numInputs):
                self.weights[nodeIn][nodeOut] -= self.weightCostGradient[nodeIn][nodeOut] * learnRate

class Network:
    def __init__(self, layerSizes):
        self.layers: list[Layer] = []
        for i in range(len(layerSizes) - 1):
            layer = Layer(layerSizes[i], layerSizes[i+1])
            self.layers.append(layer)

    def PrintNetwork(self):
        for layer in self.layers:
            layer.PrintLayer()

    def CalculateOutputs(self, inputs:list[float]):
        for layer in self.layers:
            inputs = layer.CalculateOutputs(inputs)
        return inputs
    
    def Classify(self, inputs:list[float]):
        output = self.CalculateOutputs(inputs)
        return output.index(max(output))
    
    def Cost(self, datapoint:Datapoint):
        outputs = self.CalculateOutputs(datapoint.inputs)
        cost = 0.0
        for nodeOut in range(len(outputs)):
            cost += nodeCost(outputs[nodeOut], datapoint.expectedOutput[nodeOut])
        return cost

    def AverageCost(self, datapoints:list[Datapoint]):
        totalCost = 0.0
        for datapoint in datapoints:
            totalCost += self.Cost(datapoint)
        return totalCost / len(datapoints)
    
    def Learn(self, trainingSet:list[Datapoint], learnRate:float):
        h = 0.00001
        initalCost = self.AverageCost(trainingSet)
        for layer in self.layers:
            for nodeIn in range(layer.numInputs):
                for nodeOut in range(layer.numOutputs):
                    layer.weights[nodeIn][nodeOut] += h
                    deltaCost = self.AverageCost(trainingSet) - initalCost
                    layer.weights[nodeIn][nodeOut] -= h
                    layer.weightCostGradient[nodeIn][nodeOut] += deltaCost / h

            for bias in range(len(layer.bias)):
                layer.bias[bias] += h
                deltaCost = self.AverageCost(trainingSet) - initalCost
                layer.bias[bias] -= h
                layer.biasCostGradient[bias] = deltaCost / h
            layer.ApplyGradient(learnRate)

    def write_weights_bias_to_file(self, file_path: str):
        with open(file_path, 'w') as file:
            for layer in self.layers:
                file.write(f"Layer {self.layers.index(layer) + 1}\n")
                file.write("Weights:\n")
                for row in layer.weights:
                    file.write(' '.join(str(w) for w in row))
                    file.write('\n')
                file.write("Biases:\n")
                file.write(' '.join(str(b) for b in layer.bias))
                file.write('\n')

    def read_weights_bias_from_file(self, file_path: str):
        with open(file_path, 'r') as file:
            layer_index = 0
            for line in file:
                line = line.strip()
                if line.startswith("Layer"):
                    layer_index += 1
                elif line == "Weights:":
                    weights = []
                    for _ in range(self.layers[layer_index - 1].numInputs):
                        row = [float(value) for value in file.readline().strip().split()]
                        weights.append(row)
                    self.layers[layer_index - 1].weights = weights
                elif line == "Biases:":
                    biases = [float(value) for value in file.readline().strip().split()]
                    self.layers[layer_index - 1].bias = biases

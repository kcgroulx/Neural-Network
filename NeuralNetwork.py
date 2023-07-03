import numpy as np

def Activation(weightedOutput:float):
    return 1 / (1 + np.exp(-weightedOutput))

class Layer:
    def __init__(self , numInputs:int, numOutputs:int):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.weights = [[1.0 for _ in range(numOutputs)] for _ in range(numInputs)]
        self.bias = [0] * numOutputs
    
    def CalculateOutputs(self, inputs:list[float]):
        weightedOutputs = [0.0] * self.numOutputs
        for nodeOut in range(self.numOutputs):
            weightedOutput = self.bias[nodeOut]
            for nodeIn in range(self.numInputs):
                weightedOutput += inputs[nodeIn] * self.weights[nodeIn][nodeOut]
            weightedOutputs[nodeOut] = Activation(weightedOutput)
        return weightedOutputs

class Network:
    def __init__(self, layerSizes):
        self.layers: list[Layer] = []
        for i in range(len(layerSizes) - 1):
            layer = Layer(layerSizes[i], layerSizes[i+1])
            self.layers.append(layer)
            
    def CalculateOutputs(self, inputs:list[float]):
        if(len(inputs) != self.layers[0].numInputs):
            return "Invalid Input"
        for layer in self.layers:
            inputs = layer.CalculateOutputs(inputs)
        return inputs
    
    def Classify(self, inputs:list[float]):
        output = self.CalculateOutputs(inputs)
        return output.index(max(output))
    
class Datapoint:
    def __init__(self, label, coordinates:list[float]):
        self.label = label
        self.coordinates = coordinates

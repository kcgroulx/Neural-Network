import random
import math
import io

def Activation(weightedOutput:float):
        return (1.0 / (1.0 + math.exp(-weightedOutput)))

def nodeCost(outputActivation:float, expectedOutput:float):
       error = outputActivation - expectedOutput
       return (error * error)

def get_random_sample(datapoint_list, sample_size):
    if sample_size >= len(datapoint_list):
        return datapoint_list  # Return the entire list if the sample size is equal to or larger than the list size
    else:
        indices = random.sample(range(len(datapoint_list)), sample_size)
        return [datapoint_list[i] for i in indices]

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
        i = 0
        string_output = io.StringIO()
        for layer in self.layers:
            print("Layer:", i, file=string_output)
            print("Weights:", layer.weights, file=string_output)
            print("Biases:", layer.bias, file=string_output)
        NetworkString = string_output.getvalue()
        string_output.close()
        return NetworkString
            

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
        h = 0.000001
        initalCost = self.AverageCost(trainingSet)
        for layer in self.layers:
            for nodeIn in range(layer.numInputs):
                for nodeOut in range(layer.numOutputs):
                    temp = layer.weights[nodeIn][nodeOut]
                    layer.weights[nodeIn][nodeOut] += h
                    deltaCost = self.AverageCost(trainingSet) - initalCost
                    layer.weights[nodeIn][nodeOut] = temp
                    layer.weightCostGradient[nodeIn][nodeOut] = deltaCost / h
            for bias in range(len(layer.bias)):
                temp = layer.bias[bias]
                layer.bias[bias] += h
                deltaCost = self.AverageCost(trainingSet) - initalCost
                layer.bias[bias] = temp
                layer.biasCostGradient[bias] = deltaCost / h
        self.ApplyGradients(learnRate)
    
    def ApplyGradients(self, learnRate):
        for layer in self.layers:
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




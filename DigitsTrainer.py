import NeuralNetwork as NN
import ReadIDX
import Matrix

# Load image and label data
images = ReadIDX.read_idx_images('train-images.idx3-ubyte')
labels = ReadIDX.read_idx_labels('train-labels.idx1-ubyte')

network = NN.Network([784,200,10])
network.ReadNetworkFromFile('digitsNetwork.txt')

datapoints = []
testingData = []


print("Loading dataset...")
# Gets dataset
for index in range(0, len(images) - 3000):
    expectedOutput = []
    inputs = []
    image = Matrix.RandomImageTransform(images[index], 15, 3)
    for _ in range(10):
        expectedOutput.append(0.0)
    expectedOutput[ int(labels[index]) ] = 1.0

    for row in range(0, 28):
        for col in range(0,28):
            inputs.append( float(image[row][col]) / 255 )
    datapoint = NN.Datapoint(expectedOutput, inputs)
    datapoints.append(datapoint)

# Gets testingDataset
for index in range(57000, 60000):
    expectedOutput = []
    inputs = []
    image = images[index]
    for _ in range(10):
        expectedOutput.append(0.0)
    expectedOutput[ int(labels[index]) ] = 1.0

    for row in range(0, 28):
        for col in range(0,28):
            inputs.append( float(image[row,col]) / 255 )
    datapoint = NN.Datapoint(expectedOutput, inputs)
    testingData.append(datapoint)

print("Dataset Loaded")
print("Learning...")

learnrate = 0.1
regularization = 0.0
momentum = 0.9
for j in range(0, 100):
    for i in range(0, 10):
        sample = NN.get_random_sample(datapoints, int(len(datapoints) / 1000))
        network.Learn(sample, learnrate, regularization, momentum)
    print(f"{j} Accuracy: {network.GetAccuracy(testingData):.4f} ")
    network.WriteNetworkToFile('digitsNetwork.txt')

print("Done Learning")
    


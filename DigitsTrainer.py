import NeuralNetwork as NN
import ReadIDX
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import random

# Load image and label data
images = ReadIDX.read_idx_images('train-images.idx3-ubyte')
labels = ReadIDX.read_idx_labels('train-labels.idx1-ubyte')

network = NN.Network([784,100,10])

#network.ReadNetworkFromFile('digitsNetwork.txt')

datapoints = []
testingData = []

def display_image(array):
    plt.imshow(array, cmap='gray')  # Use 'gray' colormap for grayscale images
    plt.axis('off')  # Turn off axis
    plt.show()

def apply_transformation(array, rotation_angle, offset, noise_level):
    # Apply rotation
    rotated_array = rotate(array, rotation_angle, reshape=False, mode='constant', cval=0)
    # Apply offset
    offset_array = rotated_array + offset
    # Apply random noise
    noise = np.random.normal(scale=noise_level, size=array.shape)
    noisy_array = offset_array + noise
    return noisy_array

# Gets dataset
for index in range(0, len(images) - 3000):
    expectedOutput = []
    inputs = []
    image = apply_transformation(images[index], random.uniform(-30,30), random.uniform(-10,10), 5)
    for _ in range(10):
        expectedOutput.append(0.0)
    expectedOutput[ int(labels[index]) ] = 1.0

    for row in range(0, 28):
        for col in range(0,28):
            inputs.append( float(image[row,col]) / 255 )
    datapoint = NN.Datapoint(expectedOutput, inputs)
    datapoints.append(datapoint)

# Gets dataset
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
for j in range(0, 100):
    for i in range(0, 100):
        sample = NN.get_random_sample(datapoints, int(len(datapoints) / 500))
        network.Learn(sample, 1)
    print(f"Accuracy: {network.GetAccuracy(testingData):.4f}")
    network.WriteNetworkToFile('digitsNetwork.txt')

print("Done Learning")
    


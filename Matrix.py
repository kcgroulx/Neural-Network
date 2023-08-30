import math
import random
import matplotlib.pyplot as plt

def RotatePoint(point, angle):
    x, y = point
    new_x = x * math.cos(angle) - y * math.sin(angle)
    new_y = x * math.sin(angle) + y * math.cos(angle)
    return new_x, new_y

def RotateMatrix(matrix, angleDeg):
    angelRad = math.radians(angleDeg)
    rows = len(matrix)
    cols = len(matrix[0])
    center_x = cols / 2
    center_y = rows / 2
    rotated_matrix = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        for col in range(cols):
            x = col - center_x
            y = row - center_y
            new_x, new_y = RotatePoint((x, y), angelRad)
            new_x += center_x
            new_y += center_y

            # Perform linear interpolation to approximate rotated pixel value
            x1, y1 = int(new_x), int(new_y)
            x2, y2 = x1 + 1, y1 + 1

            if 0 <= x1 < cols - 1 and 0 <= y1 < rows - 1:
                interpolated_value = (
                    (x2 - new_x) * (y2 - new_y) * matrix[y1][x1] +
                    (new_x - x1) * (y2 - new_y) * matrix[y1][x2] +
                    (x2 - new_x) * (new_y - y1) * matrix[y2][x1] +
                    (new_x - x1) * (new_y - y1) * matrix[y2][x2]
                )
                rotated_matrix[row][col] = int(interpolated_value)
    return rotated_matrix

def ShiftMatrix(matrix, shift_x, shift_y):
    rows = len(matrix)
    cols = len(matrix[0])
    
    shifted_matrix = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        for col in range(cols):
            new_row = row + shift_y
            new_col = col + shift_x

            if 0 <= new_row < rows and 0 <= new_col < cols:
                shifted_matrix[new_row][new_col] = matrix[row][col]

    return shifted_matrix

def ApplyImageTransform(matrix, angle, shift_x, shift_y):
    rotated_matrix = RotateMatrix(matrix, angle)
    return ShiftMatrix(rotated_matrix, shift_x, shift_y)

def RandomImageTransform(matrix, angelRange, shiftRange):
    randomAngle = random.randint(-angelRange, angelRange)
    randomxShift = random.randint(-shiftRange, shiftRange)
    randomyShift = random.randint(-shiftRange, shiftRange)
    return ApplyImageTransform(matrix, randomAngle, randomxShift, randomyShift)


def display_image(array):
    plt.imshow(array, cmap='gray')  # Use 'gray' colormap for grayscale images
    plt.axis('off')  # Turn off axis
    plt.show()
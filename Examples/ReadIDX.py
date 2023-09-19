import struct
import numpy as np

# Function to read image data
def read_idx_images(filename):
    with open(filename, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_items = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        
        images = []
        for _ in range(num_items):
            image_data = struct.unpack('B' * (num_rows * num_cols), f.read(num_rows * num_cols))
            images.append(list(image_data))
            
        # Reshape images to (28, 28)
        images = [np.array(image).reshape((num_rows, num_cols)) for image in images]
        
        return images

# Function to read label data
def read_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_items = struct.unpack('>I', f.read(4))[0]
        
        labels = struct.unpack('B' * num_items, f.read(num_items))
        return labels
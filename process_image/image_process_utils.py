import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import random
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

##Useful functions

def find_multiple_lines(filename):
    # Load the existing sample image
    relevant_array, relevant_array_size, firstblack, lastblack, image_array = prep_image(filename)

    # Detect and remove all lines
    lines, blank_image = detect_and_process_lines(image_array)

    return lines, blank_image, image_array

def detect_and_process_lines(image):
    lines = []  # Array to store detected lines
    current_image = np.copy(image)  # Start with the original image
    lines_to_find = True

    while lines_to_find:
        # Run the find_and_remove_black_line function
        line_indices, current_image = find_and_remove_black_line(current_image)

        # If no line is found, break the loop
        if len(line_indices) > 0:
            lines.append(line_indices)
        else:
            lines_to_find = False

    return lines, current_image

def find_and_remove_black_line(image):
    rows, cols = image.shape
    foundline = False
    checkline = False
    line_indices = []
    temporary_line_indices = []
    last_column_black_pixels = []

    for col in range(cols):
        temporary_line_indices = []  # Reset for the current column
        added_to_line = False  # Track whether we add to the line in this column
    
        if foundline:
            checkline = True
    
        for row in range(rows - 1, -1, -1):  # Traverse from bottom to top
            value = image[row, col]
    
            if value < 40:  # Black pixel found
                if not checkline:  # This is the first line ever found
                    foundline = True
                    line_indices.append((row, col))
                    last_column_black_pixels.append(row)
                elif checkline:  # If it's a secondary line, check if it's part of the current line
                    temporary_line_indices.append(row)
            else:  # Non-black pixel
                if checkline:  # If we're checking for a new line
                    # Check if temporary indices match any previous line indices
                    if any(r in last_column_black_pixels for r in temporary_line_indices):
                        # Extend the line
                        for r in temporary_line_indices:
                            line_indices.append((r, col))
                        last_column_black_pixels = temporary_line_indices[:]
                        added_to_line = True  # Mark that we added to the line
                        break
                    else:  # If no match, reset temporary indices
                        temporary_line_indices = []
    
        # End the loop prematurely if checkline is enabled and no new pixels were added
        if checkline and not added_to_line:
            break

    # Create modified image with the line removed
    modified_image = np.copy(image)
    for row, col in line_indices:
        modified_image[row, col] = 255  # Set the pixels to white (or non-black)

    return line_indices, modified_image

def generate_array_from_line(line, size=100):
    # Create a blank 2D array (white background)
    array = np.ones((size, size), dtype=np.uint8) * 255
    # Draw the line (black pixels)
    for row, col in line:
        array[row, col] = 0  # Set the pixel to black
    return array

def process_array(image_array):
    output, firstblack, lastblack = hightest_black_in_array(image_array)
    relevant_array = output[firstblack:lastblack]
    relevant_array_size = lastblack - firstblack
    return relevant_array, relevant_array_size, firstblack, lastblack

def hightest_black_in_array(image):
    rows, cols = image.shape
    output = np.ones(cols)*(-1)
    firstblack = cols
    lastblack = 0
    for col in range(cols):  # Iterate over columns
        for row in range(rows):  # Iterate from top to bottom
            value = image[row, col]
            if value < 40:
                output[col] = row
                if col < firstblack:
                    firstblack = col
                if col > lastblack:
                    lastblack = col
                break  # Stop checking further rows in this column
    return output, firstblack, lastblack

def calibrate_array(array, image_size):
    # Check if array is empty
    if len(array) == 0:
        raise ValueError("Input array is empty. Cannot calibrate.")
    # Remove NaN and Inf values
    array = array[~np.isnan(array)]  # Remove NaN
    array = array[np.isfinite(array)]  # Remove Inf
    # Compute the mean of the array
    mean_value = np.mean(array)
    # Ensure mean_value is a finite number
    if not np.isfinite(mean_value):
        raise ValueError("Mean value is NaN or Inf. Check input array.")
    # Round the mean value to the nearest integer
    meany = int(round(mean_value))
    # Subtract the mean (meany) from the array
    centered_array = array - meany
    # Rescale the array to range [-1, 1] using half the image size
    half_image_size = image_size / 2
    rescaled_array = centered_array / half_image_size

    return rescaled_array, meany

def chunk_and_pad_array(array, A):
    n_chunks = len(array) // A
    remainder = len(array) % A
    chunks = [array[i * A:(i + 1) * A] for i in range(n_chunks)]
    if remainder > 0:
        last_chunk = array[-remainder:]
        padding_value = last_chunk[-1]
        padded_chunk = np.pad(last_chunk, (0, A - remainder), constant_values=padding_value)
        chunks.append(padded_chunk)
    return chunks

def test_model_on_chunk(chunk_array, model):
    input_tensor = torch.from_numpy(chunk_array).float().unsqueeze(0)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    return output_tensor.squeeze(0).cpu().numpy()

def stitch_and_truncate_chunks(chunks, A):
    stitched_array = np.concatenate(chunks)
    return stitched_array[:A]

def uncalibrate_array(array, meany, image_size):
    half_image_size = image_size / 2
    rescaled_array = array * half_image_size
    return rescaled_array + meany

def pad_array_with_neg_ones(array, left_pad, right_pad):
    return np.pad(array, (left_pad, right_pad), mode='constant', constant_values=-1)

def array_to_image(output_array, line_height, fill_value=0):
    output_array = np.array(output_array, dtype=int)
    cols = len(output_array)
    image = np.full((cols, cols), 255, dtype=int)
    for col, row in enumerate(output_array):
        if row >= 0:
            start_row = max(0, row - line_height + 1)
            image[start_row:row + 1, col] = fill_value
    return image

def combine_output_images(output_images):
    # Start with a blank white image (255) of the same size as the output images
    combined_image = np.ones_like(output_images[0]) * 255

    # Combine all images: retain black pixels (0) from any image
    for image in output_images:
        combined_image = np.minimum(combined_image, image)

    return combined_image

## Unused Functions

def prep_image(filename):
    image_array = transparent_img_to_array(filename)
    output, firstblack, lastblack = hightest_black_in_array(image_array)
    relevant_array = output[firstblack:lastblack]
    relevant_array_size = lastblack - firstblack
    return relevant_array, relevant_array_size, firstblack, lastblack, image_array

def transparent_img_to_array(filename):
    image = Image.open(filename)
    # Ensure the image is in RGBA mode
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
        
    white_background = Image.new('RGBA', image.size, (255, 255, 255, 255))
    blended_image = Image.alpha_composite(white_background, image)
    gray_image = blended_image.convert('L')
    image_array = np.array(gray_image)
    return image_array

def random_increase_chunk(chunk):
    random_increments = np.random.randint(0, 101)
    return chunk + random_increments

def center_and_rescale(array, height=500):
    center = height / 2.0
    return (array - center) / height

def reverse_center_and_rescale(array, height=500):
    center = height / 2.0
    return (array * height) + center

def output_test(modified_chunks, relevant_array_size, firstblack, lastblack, image_array):
    processed_array = stitch_and_truncate_chunks(modified_chunks, relevant_array_size)
    output_array = pad_array_with_neg_ones(processed_array, firstblack, len(image_array) - lastblack)
    return array_to_image(output_array, 3)

def model_on_image(model, filename, chunk_size = 500):
    image_size = 500
    lines, blank_image, initial_image_array = find_multiple_lines(filename)
    # Initialize a list to store output image arrays
    output_images = []
    for i, line in enumerate(lines):
        # Generate an image array for the line
        image_array = generate_array_from_line(line, size=image_size)
        relevant_array, relevant_array_size, firstblack, lastblack = process_array(image_array)  # Process the array
    
        # Recalibrate the array to center for better ML performance
        rescaled_array, meany = calibrate_array(relevant_array, image_size)
    
        # Chunk, modify, and reconstruct the array
        chunks = chunk_and_pad_array(rescaled_array, chunk_size)
        modified_chunks = [test_model_on_chunk(chunk, model) for chunk in chunks]
        stitched_array = stitch_and_truncate_chunks(modified_chunks, relevant_array_size)
        processed_array = uncalibrate_array(stitched_array, meany, image_size)
    
        # Generate the padded output array and image
        output_array = pad_array_with_neg_ones(processed_array, firstblack, len(image_array) - lastblack)
        output_image_array = array_to_image(output_array, 3)
    
        # Append the current output image array to the list
        output_images.append(output_image_array)
    
    # Combine all output image arrays into a single array
    final_output_image = combine_output_images(output_images)
    return initial_image_array, final_output_image

class NN_AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(NN_AutoEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # Fully connected layer
            nn.Linear(128, 128),  # Fully connected layer
            nn.Linear(128, input_dim),  # Fully connected layer
        )

    def forward(self, x):
        return self.fc(x)

class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim):
        super(FullyConnectedNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim)  # Output layer
        )

    def forward(self, x):
        return self.fc(x)


def load_model(model_class, model_path, input_dim):
    """
    Loads a model of the specified class with the given input dimension and path.

    Parameters:
    - model_class (nn.Module): The class of the model to be loaded.
    - model_path (str): Path to the saved model state.
    - input_dim (int): Input dimension for the model.

    Returns:
    - model: Loaded PyTorch model in evaluation mode.
    """
    model = model_class(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def get_model(model_id, input_dim=500):
    """
    Returns a pre-trained model based on the model ID.

    Parameters:
    - model_id (int): Integer specifying the model to load.
                      1 - FullyConnectedNN (No Noise)
                      2 - NN_AutoEncoder (No Noise)
                      3 - FullyConnectedNN (With Noise)
                      4 - NN_AutoEncoder (With Noise)
    - input_dim (int): Input dimension for the model (default: 500).

    Returns:
    - model: The loaded PyTorch model.
    """
    model_paths = {
        1: "process_image/noiseless_model500.pth",          # FullyConnectedNN (No Noise)
        2: "process_image/autoenc128.pth",                  # NN_AutoEncoder (No Noise)
        3: "process_image/noise_model500.pth",              # FullyConnectedNN (With Noise)
        4: "process_image/noiseautoenc_2layer_128.pth",     # NN_AutoEncoder (With Noise)
    }

    model_classes = {
        1: FullyConnectedNN,
        2: NN_AutoEncoder,
        3: FullyConnectedNN,
        4: NN_AutoEncoder,
    }

    if model_id not in model_paths:
        raise ValueError(f"Invalid model_id: {model_id}. Must be one of {list(model_paths.keys())}.")

    model_path = model_paths[model_id]
    model_class = model_classes[model_id]

    # Load the specified model
    return load_model(model_class, model_path, input_dim)

def process_image(filename, model_number, output_name):
    chunk_size = 500
    model = get_model(model_number)
    initial_image_array, final_output_image = model_on_image(model, filename)
    Image.fromarray(final_output_image.astype(np.uint8)).save(output_name)
    return True

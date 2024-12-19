from PIL import Image
import numpy as np
import cv2

img_path = '/YOUR/PATH/IMAGE.png'
img = cv2.imread(img_path, 0) # read image as grayscale. Set second parameter to 1 if rgb is required 


def scan_image_for_heights(image_array, threshold=50):
    """
    Scans an image array for the first sufficiently dark pixel in each column.

    Parameters:
        image_array (numpy.ndarray): The input image array (H x W for grayscale or H x W x C for color).
        threshold (int): Pixel intensity threshold for "sufficiently dark" (0 is black, 255 is white).

    Returns:
        list: A 1D array where each element corresponds to the height (row index) of the first
              sufficiently dark pixel in that column, or -1 if no such pixel is found.
    """
    # Ensure the image is grayscale
    if len(image_array.shape) == 3:  # If the image has multiple channels (e.g., RGB)
        image_array = np.mean(image_array, axis=2)  # Convert to grayscale by averaging channels

    height, width = image_array.shape
    heights = [-1] * width  # Initialize the 1D array with -1 for each column

    for col in range(width):
        for row in range(height):
            if image_array[row, col] < threshold:  # Check if pixel is "sufficiently dark"
                heights[col] = row
                break  # Stop scanning the column once the first dark pixel is found

    return heights

# Example usage
image_path = 'static/saved_image.png'
processed_path = 'process_image/processed_image.png'

# Convert image to array
image_array = imread(image_path)
print(type(image_array))
print("image_array:", image_array)

# Call the function with a threshold of 100
heights = scan_image_for_heights(image_array, threshold=0.5)

print("1D Heights Array:", heights)

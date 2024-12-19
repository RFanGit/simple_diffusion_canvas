import os
import random
from PIL import Image, ImageDraw

# Define constants
OUTPUT_DIR = "generated_images"
IMAGE_SIZE = (100, 100)  # Size of each image (width, height)
SETS = ["set1", "set2", "set3"]  # Image sets
IMAGES_PER_SET = 4  # Number of images per set
PLACEHOLDER_IMAGES = 4  # Number of placeholder images

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
for set_name in SETS:
    os.makedirs(os.path.join(OUTPUT_DIR, set_name), exist_ok=True)

# Function to generate an image with random circles and squares
def generate_image(save_path):
    image = Image.new("RGB", IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(image)

    # Randomly draw shapes
    for _ in range(5):  # Add 5 random shapes
        shape_type = random.choice(["circle", "square"])
        x1, y1 = random.randint(0, IMAGE_SIZE[0] - 20), random.randint(0, IMAGE_SIZE[1] - 20)
        x2, y2 = x1 + random.randint(10, 30), y1 + random.randint(10, 30)

        if shape_type == "circle":
            draw.ellipse([x1, y1, x2, y2], fill=random.choice(["red", "blue", "green", "yellow"]))
        elif shape_type == "square":
            draw.rectangle([x1, y1, x2, y2], fill=random.choice(["red", "blue", "green", "yellow"]))

    # Save the image
    image.save(save_path)

# Generate placeholder images
for i in range(PLACEHOLDER_IMAGES):
    generate_image(os.path.join(OUTPUT_DIR, f"placeholder{i+1}.png"))

# Generate images for each set
for set_name in SETS:
    for i in range(IMAGES_PER_SET):
        generate_image(os.path.join(OUTPUT_DIR, set_name, f"{set_name}_image{i+1}.png"))

print(f"Images have been generated in the '{OUTPUT_DIR}' directory.")

const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const saveBtn = document.getElementById('saveBtn');
const imageDisplay = document.getElementById('imageDisplay');
const imageSelector = document.getElementById('imageSelector');
const mainImage = document.getElementById('mainImage');

let drawing = false;

// Adjust canvas dimensions for drawing
canvas.width = 500;
canvas.height = 500;

// Set up initial drawing styles
ctx.lineWidth = 2; // Thickness of the line
ctx.lineCap = 'round'; // Smooth line ends
ctx.strokeStyle = 'black'; // Line color

// Start drawing
canvas.addEventListener('mousedown', (event) => {
  drawing = true;
  const { x, y } = getMousePos(event);
  ctx.beginPath(); // Start a new path
  ctx.moveTo(x, y); // Move to the current mouse position
});

// Draw on the canvas
canvas.addEventListener('mousemove', (event) => {
  if (!drawing) return;
  const { x, y } = getMousePos(event);
  ctx.lineTo(x, y); // Draw a line to the new position
  ctx.stroke(); // Render the line
});

// Stop drawing
canvas.addEventListener('mouseup', () => {
  drawing = false;
  ctx.closePath(); // Finish the current path
});

// Clear the canvas
clearBtn.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the entire canvas
});

saveBtn.addEventListener('click', () => {
    const canvas = document.getElementById('drawCanvas');
    const dataURL = canvas.toDataURL('image/png');

    // Get the selected model value from the dropdown
    const modelSelector = document.getElementById('imageSelector');
    const selectedModel = modelSelector.value;

    fetch('/save', {
        method: 'POST',
        body: JSON.stringify({
            image: dataURL, 
            model: selectedModel  // Include the selected model in the payload
        }),
        headers: {
            'Content-Type': 'application/json',
        },
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.path) {
                // Update the image's src to display the saved image
                imageDisplay.src = data.path + '?t=' + new Date().getTime();
            }
            //alert(data.message);
        })
        .catch((error) => console.error('Error:', error));
});

// Helper function to get mouse position relative to the canvas
function getMousePos(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

// Define image sets with associated text descriptions
const imageSets = {
    set1: {
        src: "/static/images/NN_AutoEncoder_(With_Noise).png",
        alt: "Autoencoder with Noise",
        description: "This model denoises drawings by applying an autoencoder trained with noisy input data. Although it does not perfectly fit my training data, it has learned general principles for smoothening lines."
    },
    set2: {
        src: "/static/images/NN_AutoEncoder.png",
        alt: "Autoencoder",
        description: "This model uses an autoencoder to improve drawings. It has great performance on my training data-set, but does poorly in practice."
    },
    set3: {
        src: "/static/images/FullyConnectedNN_(With_Noise).png",
        alt: "Simple Neural Net with Noise",
        description: "This model applies a simple neural network trained with noisy input data to enhance drawings."
    },
    set4: {
        src: "/static/images/FullyConnectedNN.png",
        alt: "Simple Neural Net",
        description: "This model uses a simple neural network to enhance drawings. It has great performance on my training data-set, but does poorly in practice."
    }
};

// Get references to the necessary elements
const imageDescription = document.getElementById('imageDescription');

// Function to update image and description
function updateGallery(value) {
    const selectedSet = imageSets[`set${value}`];
    if (selectedSet) {
        mainImage.src = selectedSet.src; // Update the image source
        mainImage.alt = selectedSet.alt; // Update the alt text
        imageDescription.textContent = selectedSet.description; // Update the description text
    }
}

// Set the default option to 1 on page load
document.addEventListener('DOMContentLoaded', () => {
    imageSelector.value = "1"; // Set dropdown to option 1
    updateGallery("1"); // Update the image and description for option 1
});

// Update the image and text when the selector changes
imageSelector.addEventListener('change', (event) => {
    updateGallery(event.target.value);
});
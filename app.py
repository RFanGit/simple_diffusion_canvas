from flask import Flask, render_template, request, jsonify
import base64
from process_image.image_process_utils import process_image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/save', methods=['POST'])
def save():
    data = request.get_json()
    image_data = data.get('image')
    model_selection = data.get('model')  # Get the model selection from the request
    print (model_selection)

    if not image_data:
        return jsonify({'message': 'No image data received!'}), 400

    if not model_selection:
        return jsonify({'message': 'No model selected!'}), 400

    try:
        # Decode the base64 image data and save it
        header, encoded = image_data.split(',', 1)
        binary_data = base64.b64decode(encoded)
        saved_path = 'static/saved_image.png'
        processed_path = 'static/output_image.png'
        with open(saved_path, 'wb') as f:
            f.write(binary_data)

        model_choice = 1
        if model_selection == "1":
            model_choice = 4
        if model_selection == "2":
            model_choice = 2
        if model_selection == "3":
            model_choice = 3

        # Call the image_process function using the selected model value
        process_image(saved_path, model_choice, processed_path)

        # Return the path to the processed image
        return jsonify({
            'message': 'Image saved and processed successfully!',
            'path': '/' + processed_path
        })
    except Exception as e:
        return jsonify({'message': f'Error saving or processing image: {str(e)}'}), 500

@app.route('/get-selected-option', methods=['POST'])
def get_selected_option():
    selected_value = request.json.get('selectedValue')
    print(f"Selected option value: {selected_value}")
    return jsonify({"status": "success", "selectedValue": selected_value})

if __name__ == '__main__':
    app.run(debug=True)
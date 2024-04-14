from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from flask_cors import CORS
from keras.preprocessing import image

app = Flask(__name__)
CORS(app)
# Load the trained models
COTTON_MODEL = tf.keras.models.load_model("models/cotton_api.h5")
CORN_MODEL = tf.keras.models.load_model("models/corn_api.h5")
POTATO_MODEL = tf.keras.models.load_model("models/potato_api.h5")
RICE_MODEL = tf.keras.models.load_model("models/rice_api.h5")
TOMATO_MODEL = tf.keras.models.load_model("models/tomato_api.h5")

# Image preprocessing function
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.get('/')
def index():
    return 'Hello, welcome to the Plant Disease Detection API!'

@app.route('/predict/cotton', methods=['POST'])
def predict_cotton():
    return predict(COTTON_MODEL, {
        0: 'Aphids', 1: 'Army_worm', 2: 'Bacterial_blight',
        3: 'Healthy', 4: 'Powdery_mildew', 5: 'Target_spot'
    })

@app.route('/predict/corn', methods=['POST'])
def predict_corn():
    return predict(CORN_MODEL, {
        0: 'blight', 1: 'common_rust', 2: 'grey_leaf_spot',
        3: 'healthy', 4: 'rust', 5: 'scab'
    })

@app.route('/predict/potato', methods=['POST'])
def predict_potato():
    return predict(POTATO_MODEL, {
        0: 'Potato_Early_blight', 1: 'Potatohealthy', 2: 'Potato_Late_blight'
    })

@app.route('/predict/rice', methods=['POST'])
def predict_rice():
    return predict(RICE_MODEL, {
        0: 'Bacterialblight', 1: 'Blast', 2: 'Brownspot', 3: 'Tungro'
    })

@app.route('/predict/tomato', methods=['POST'])
def predict_tomato():
    return predict(TOMATO_MODEL, {
        0: 'Tomato_healthy', 1: 'Tomato_Late_blight', 2: 'Tomato_Leaf_Mold',
        3: 'Tomato_Septoria_leaf_spot', 4: 'Tomato_Spider_mites_Two_spotted_spider_mite',
        5: 'Target_spot'
    })

def predict(model, class_indices):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Save the uploaded file
    file_path = 'uploaded_image.jpg'
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Make predictions using the loaded model
    predictions = model.predict(img_array)

    # Get the predicted class
    predicted_class = class_indices[np.argmax(predictions)]

    # Return the prediction as JSON
    result = {'predicted_class': predicted_class}
    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7000,debug=True)
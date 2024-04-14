# CropCare - Plant Disease Detection

This project implements a Plant Disease Detection System using machine learning models deployed as Flask APIs. The system allows users to upload images of plants and receive predictions about possible diseases affecting the plants. It includes separate models for detecting diseases in cotton, corn, potato, rice, and tomato plants.

## Description

CropCare is an advanced tool that uses Artificial Intelligence (AI) and image processing to detect crop diseases quickly. CropCare gives reliable diagnoses for common crops such as corn, cotton, rice, potato, and tomato by analyzing smartphone photographs of the fields. This technology saves farmers money, allows them to quickly access new agricultural instruments, and protects their livelihoods by reducing crop losses. Overall, CropCare's technology has the potential to transform farming techniques, ensuring Indian agriculture's sustainability and productivity.

## How It Works

1. **Select Plant Type**: Users can choose the type of plant for which they want to detect diseases from a dropdown menu on the web interface.

2. **Upload Image**: Users can upload an image of the selected plant using the provided file input field.

3. **Prediction**: After uploading the image, users can click the "Predict" button to send the image to the corresponding Flask API endpoint based on the selected plant type.

4. **Result Display**: The Flask API processes the image using the appropriate machine learning model and returns the predicted disease class. The result is then displayed to the user on the web interface.

**Try here**: [CropCare](https://cropcare-du.vercel.app/) <sub>use images from the demo images folder if needed</sub>.

## Tech Stack
 - Machine Learning
 - Tensorflow
 - Flask
 - JavaScript
 - HTML\CSS

## Project Dependencies

- Flask: Used for building the backend server and API endpoints.
- Flask-CORS: Enables Cross-Origin Resource Sharing (CORS) for handling requests from the front end.
- TensorFlow: Framework used for training and deploying machine learning models.
- Keras: High-level neural networks API, used for building and training deep learning models.
- NumPy: Library for numerical computing, used for array manipulation and preprocessing images.
- JavaScript: Used for client-side scripting to handle user interactions on the web interface.

## Setup Instructions

1. **Clone the repository**: Clone the repository to your local machine:
```git clone https://github.com/himavarshithreddy/cropcare.git```
2. **Install dependencies**: Install project dependencies using
```pip install -r requirements.txt```

3. **Run the Flask server**: Run the Flask server locally:
```python api.py```

4. **Access the web interface**: Navigate to the web directory and open index.html in your browser.

5. **Detect diseases**: Upload an image of a plant, select the plant type, and click the "Predict" button to detect diseases.

## Video Demo
https://youtu.be/U99cPbgq7js

## Additional Information

- **Model Training**: The machine learning models used in this project were trained on labelled datasets of images of diseased and healthy plants. The datasets were preprocessed and split into training and validation sets to train the models.

- **Performance**: The accuracy of the models may vary depending on factors such as the quality of the input images and the complexity of the diseases being detected. Continuous improvement and refinement of the models may be necessary to achieve higher accuracy rates.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


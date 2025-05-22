from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import os

app = Flask(__name__)


model_path = "vgg19 (1).h5"
vgg_model = load_model(model_path)


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    
    file = request.files['file']

    
    file_path = "temp.jpg"
    file.save(file_path)

    
    filename, file_extension = os.path.splitext(file.filename)

    if file_extension.lower() == '.png':
        
        os.remove(file_path)
        
        return jsonify({
            'predicted_class': 'Undefined',
            'prediction_probabilities': [0, 0, 0, 0, 1],  
        })

    
    img_array = preprocess_image(file_path)

    
    predictions = vgg_model.predict(img_array)

    
    vgg_class_labels = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

    
    predicted_class = vgg_class_labels[np.argmax(predictions)]
    prediction_probabilities = predictions[0].tolist()

    
    os.remove(file_path)

    
    return jsonify({
        'predicted_class': predicted_class,
        'prediction_probabilities': prediction_probabilities,
    })


@app.route('/')
def index():
    return render_template('pythonlast.html')

if __name__ == '__main__':
    app.run(debug=True)


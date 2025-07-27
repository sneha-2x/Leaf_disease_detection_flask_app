from flask import Flask, render_template, request, redirect, url_for
import os
import threading
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load your trained model
model = load_model('model/leaf_disease_model.h5')

# Class labels (replace with your own if different)
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# Function to delete file after delay
def delete_file_later(path, delay=5):
    def delete():
        time.sleep(delay)
        if os.path.exists(path):
            os.remove(path)
    threading.Thread(target=delete).start()

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    file = request.files['file']

    if not allowed_file(file.filename):
        return "❌ Invalid file type. Please upload a PNG or JPG image."

    # Save the uploaded file
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and preprocess the image
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalize

    # Predict
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    class_label = class_names[class_index]
    confidence = float(np.max(predictions[0]))

    # Use relative path for HTML
    relative_path = os.path.join('static', 'uploaded', filename)

    # Render the result page
    response = render_template('index.html',
                               prediction=class_label,
                               confidence=round(confidence * 100, 2),
                               img_path=relative_path)

    # Delete file after short delay
    delete_file_later(filepath, delay=10)

    return response

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

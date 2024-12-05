import os
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load the pre-trained model
model = tf.keras.models.load_model('modelHamza/finetuned_model.h5')
# Alternatively, if you saved the model in Keras format
# model = tf.keras.models.load_model('models/finetuned_model.keras')

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Preprocess the uploaded image
        image = tf.keras.preprocessing.image.load_img(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            target_size=(224, 224)
        )
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0)

        # Use the loaded model to make a prediction
        prediction = model.predict(image_array)
        # Interpret the prediction result and return a response
        class_names = ['healthy', 'bleached', 'other']
        predicted_class = class_names[tf.argmax(prediction[0])]
        return jsonify({'prediction': predicted_class})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
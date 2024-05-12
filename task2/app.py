from flask import Flask, render_template, request, redirect, url_for
import os
from transformers import pipeline
import tensorflow as tf
import numpy as np
from PIL import Image
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text_emotion', methods=['POST'])
def text_emotion():
    pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
    text = request.form['text']
    result = pipe(text)[0]
    return render_template('result.html', result=result,form_type="text")

@app.route('/audio_emotion', methods=['POST'])
def audio_emotion():


    pipe = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
      
    audio_file = request.files['audio_file']
    if audio_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        audio_file.save(file_path)
        pipe = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")

        result = pipe(file_path)[0]
        return render_template('result.html', result=result,form_type="audio")
    return redirect(url_for('index'))

# app = Flask(__name__)


# def load_model(model_path, weights_path):
#     with open(model_path, 'r') as json_file:
#         model_json = json_file.read()

#     # Load model architecture from JSON
#     model = tf.keras.models.model_from_json(model_json)
#     # Load model weights
#     model.load_weights(weights_path)
    
#     # Compile the model if necessary
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     return model

# model = load_model('model_a.json', 'model_weights.h5')

# def detect_emotion_image(image_path, model):
#     image = Image.open(image_path).convert('L')  # Convert to grayscale
#     image = image.resize((48, 48))
#     image = np.array(image)
#     image = image / 255.0
#     image = np.expand_dims(image, axis=-1)  # Add channel dimension
#     image = np.expand_dims(image, axis=0)   # Add batch dimension

#     predictions = model.predict(image)

#     emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}    
#     predicted_emotion_index = np.argmax(predictions)
#     result = emotion_labels[predicted_emotion_index]

#     return result

@app.route('/image_emotion', methods=['POST'])
def image_emotion():
    if request.method == 'POST':
        f = request.files['image_file']  
        file_path = os.path.join('uploads', f.filename)
        f.save(file_path)
        pipe = pipeline("image-classification", model="trpakov/vit-face-expression")
        result = pipe(file_path)[0]
        return render_template('result.html', result=result, form_type="image")
    
if __name__ == '__main__':
    app.run(debug=True)


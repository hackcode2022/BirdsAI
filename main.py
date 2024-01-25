from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)
with app.app_context():
    model = load_model('mon_modele')
@app.route('/', methods=['GET'])
def index():
    if request.method == "GET":
        return render_template('index.html')


@app.route('/handleImage', methods=['POST'])
def handleImage():
    if request.method == "POST":

        data = request.get_json()
        image_data = data.get('image')
        jpeg_base64_data = image_data
        jpeg_base64_data = jpeg_base64_data.replace("data:image/jpeg;base64,", "")
        # Convertir la base64 en bytes
        jpeg_bytes = base64.b64decode(jpeg_base64_data)

        # Charger l'image depuis les bytes
        image = Image.open(BytesIO(jpeg_bytes))
        target_size = (224, 224)
        image = image.resize(target_size)
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normaliser l'image

        dataCsv = pd.read_csv('birds.csv')
        test_data = np.load('test_data.npy')

        def display_image_and_prediction(model, img_array):
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            predicted_class_name = test_data[predicted_class];  # Nom de l'oiseau
            print("Nom de l'oiseau :", predicted_class_name)

            scientificName = dataCsv.loc[dataCsv["labels"] == predicted_class_name, "scientific name"].iloc[0]  # Récupération du nom scientifique de cet oiseau
            print("Nom scientifique :", scientificName)
            return predicted_class_name, scientificName

        predicted_class_name, scientificName = display_image_and_prediction(model, img_array)
        result = {'result': predicted_class_name, 'sname':scientificName}
        return jsonify(result)


@app.route('/documentation')
def documentation():
    return render_template('documentation.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81)

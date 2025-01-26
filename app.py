from flask import Flask, render_template, request, jsonify         
import cv2         
import numpy as np         
import base64         
from keras.models import load_model         
import os         
        
app = Flask(__name__)         
app.config['SECRET_KEY'] = 'your_secret_key_here'         
        
# Load the pre-trained Keras model         
model = load_model(r"C:\Users\Roshan Khatale\DSML19\Projects\Computer vision\Age and Gender detection\Flask\Age and gender.h5")         
gender_dict = {0: 'Male', 1: 'Female'}         
        
@app.route('/', methods=['GET', 'POST'])         
def index():         
    if request.method == 'POST':         
        # Get the uploaded file from the request         
        file = request.files['file']         
        
        # Save the file to the uploads directory         
        upload_dir = os.path.join(app.static_folder, 'uploads')         
        os.makedirs(upload_dir, exist_ok=True)               
        image_path = os.path.join(upload_dir, file.filename)         
        file.save(image_path)         
        
        # Read the image and make predictions         
        image = cv2.imread(image_path)         
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)         
        img = cv2.resize(img, (128, 128))         
        img = img.reshape(1, 128, 128, 1) / 255.0         
        pred = model.predict(img)         
        pred_gender = gender_dict[np.round(pred[0][0][0])]         
        pred_age = int(np.round(pred[1][0][0]))         
        
        # Return the predictions as a JSON response         
        response = {'gender': pred_gender, 'age': pred_age}         
        return jsonify(response)         
    else:         
        return render_template('index.html')         
        
if __name__ == '__main__':         
    app.run(debug=True)         
        
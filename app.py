from __future__ import division, print_function
from flask import Flask, render_template, request
import pickle
import numpy as np




app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html')


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Crop Recommendation'
    return render_template('crop_pred.html', title=title)

@ app.route('/fertilizer-recommend')
def fertilizer_recommend():
    title = 'Fertilizer Recommendation'
    return render_template('fert_pred.html', title=title)

@app.route('/disease-recommend', methods=['GET'])
def disease_recommend():
    # Main page
    return render_template('index.html')


@app.route('/crop_predict', methods=['POST'])
def crop_pred():
    data1 = request.form['N']
    data2 = request.form['P']
    data3 = request.form['K']
    data4 = request.form['Temp']
    data5 = request.form['Humid']
    data6 = request.form['Ph']
    data7 = request.form['Rain']

    model = pickle.load(open('crop_pred.pkl', 'rb'))

    pred = model.predict([[data1, data2, data3, data4, data5, data6, data7]])
    print(pred)
    return render_template('afterhome.html', data=pred)



@app.route('/fert_predict', methods=['POST'])
def fert_pred():
    data11 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']

    model = pickle.load(open('fertilizer_pred.pkl', 'rb'))

    pred = model.predict([[data11, data2, data3, data4, data5, data6, data7, data8]])
    print(pred)
    if pred == 0:
        pred = "10-26-26"
    elif pred ==1:
        pred = "14-35-14"
    elif pred == 2:
        pred = "17-17-17"
    elif pred == 3:
        pred = "20-20"
    elif pred == 4:
        pred = "28-28"
    elif pred == 5:
        pred = "DAP"
    else:
        pred = "Urea"
    return render_template('afterFert.html', data=pred)









# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app


# Model saved with Keras model.save()
MODEL_PATH ='tomato_leafDisease_pred.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="bactetria"
    elif preds==1:
        preds="early bligh"
    elif preds==2:
        preds="late blight"
    elif preds==3:
        preds="leaf mold"
    elif preds==4:
        preds="septoria"
    elif preds==5:
        preds="spider mild"
    elif preds==6:
        preds="target spot"
    elif preds==7:
        preds="yellow curl"
    elif preds==7:
        preds="mosaic virus"
    else:
        preds="healthy"
        
    
    
    return preds





@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == "__main__":
    app.run(port=5001,debug=True)

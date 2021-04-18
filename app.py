import os
import urllib.request
from flask import Flask, flash, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename
from flask import Flask
import pickle
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np


app = Flask(__name__,static_url_path='/static')
app.config['location'] = "./static"

# import all models
Breast_cancer_model = pickle.load(open('./ML models/Breast_cancer_model', 'rb'))
Breast_cancer_standard_scaler = pickle.load(open('./ML models/Breast_cancer_standard_scaler', 'rb'))

Heart_desease_model=pickle.load(open('./ML models/Heart_desease_model', 'rb'))
Heart_desease_standard_scaler=pickle.load(open('./ML models/Heart_desease_standard_scaler', 'rb'))

Diabetes_model=pickle.load(open('./ML models/Diabetes_model', 'rb'))
Diabetes_standard_scaler=pickle.load(open('./ML models/Diabetes_standard_scaler', 'rb'))
Pneumonia_model=load_model('./ML models/pneumonia_model.h5')

Stroke_model=pickle.load(open('./ML models/Stroke_model', 'rb'))
Stroke_standard_scaler=pickle.load(open('./ML models/Stroke_StandardScaler', 'rb'))

Maleria_model=load_model('./ML models/Maleria_model.h5')

liver_model=pickle.load(open('./ML models/Liver_model','rb'))

# app routes
def convert_list(data):
    data = data.strip('][').split(',')
    for i in range(0, len(data)):
        data[i] = float(data[i])
    return data


@app.route('/')
def upload_form():
    return render_template('page.html')


@app.route('/heart_disease')
def heart_disease():
    return render_template('heart_disease.html')

@app.route('/heart_disease', methods=['POST'])
def heart_disease_pred():
    data=convert_list(request.form['id'])
        
    predict_data=Heart_desease_standard_scaler.transform([data])
    output=int(Heart_desease_model.predict(predict_data))

    resp = jsonify({"result":output,'text':"dodo"})
    resp.status_code = 200
    return resp


@app.route('/Diabetes')
def Diabetes():
    return render_template('Diabetes.html')

@app.route('/Diabetes', methods=['POST'])
def Diabetes_pred():
    data=convert_list(request.form['id'])
        
    predict_data=Diabetes_standard_scaler.transform([data])
    output=int(Diabetes_model.predict(predict_data))

    resp = jsonify({"result":output})
    resp.status_code = 200
    return resp



@app.route('/Brest_cancer')
def brest_cancer():
    return render_template('Brest_cancer.html')

@app.route('/Brest_cancer', methods=['POST'])
def Brest_cancer_pred():
    data=convert_list(request.form['id'])
        
    predict_data=Breast_cancer_standard_scaler.transform([data])
    output=int(Breast_cancer_model.predict(predict_data))

    resp = jsonify({"result":output})
    resp.status_code = 200
    return resp


@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

@app.route('/Pneumonia', methods=['POST'])
def pneumonia_pred():
    if 'IMAGE' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    else:
        files = request.files.getlist('IMAGE')
        for file in files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['location'], filename))
            path=f'{app.config["location"]}/{filename}'
            img=image.load_img(path,target_size=(224,224))
            x=image.img_to_array(img)
            x=np.expand_dims(x,axis=0)
            img_data=preprocess_input(x)
            classes=Pneumonia_model.predict(img_data)
            os.remove(path)
        resp = jsonify({"Outcome":int(classes[0][1])})
        resp.status_code = 200
        return resp


@app.route('/Stroke')
def Stroke():
    return render_template('Stroke.html')

@app.route('/Stroke', methods=['POST'])
def Stroke_pred():
    data=convert_list(request.form['id'])
    predict_data=Stroke_standard_scaler.transform([data])
    output=int(Stroke_model.predict(predict_data))

    resp = jsonify({"result":output})
    resp.status_code = 200
    return resp



@app.route('/maleria')
def maleria():
    return render_template('maleria.html')

@app.route('/maleria', methods=['POST'])
def maleria_pred():
    if 'IMAGE' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    else:
        files = request.files.getlist('IMAGE')
        for file in files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['location'], filename))
            path=f'{app.config["location"]}/{filename}'
            img=image.load_img(path,target_size=(224,224))
            x=image.img_to_array(img)
            x=np.expand_dims(x,axis=0)
            img_data=preprocess_input(x)
            prediction=Maleria_model.predict(img_data)
            print(prediction)
            prediction=np.argmax(prediction, axis=1)
            os.remove(path)
        resp = jsonify({"Outcome":int(prediction)})
        resp.status_code = 200
        return resp


@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/liver', methods=['POST'])
def liver_pred():
    data=convert_list(request.form['id'])
    output=int(liver_model.predict([data]))

    resp = jsonify({"result":output})
    resp.status_code = 200
    return resp



if __name__ == "__main__":
    app.run(debug=True)
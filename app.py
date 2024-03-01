from flask import Flask,request,app,render_template,Response
import pandas as pd
import numpy as np
# from sklearn import 
import pickle

app = Flask(__name__)

scaler = pickle.load(open("model/StandardScale.pkl",'rb'))
model = pickle.load(open("model/modelForPredictin.pkl",'rb'))


@app.route("/")
def index():
    return render_template('index.html')
    
@app.route('/diabeties',methods=['GET','POST'])
def predict_datapoint():
    result=""
    if request.method=="POST":
        pregnancies=int(request.form.get('pregnancies'))
        glucose = float(request.form.get('glucose'))
        bloodPressure=float(request.form.get('bloodPressure'))
        skinThickness=float(request.form.get('skinThickness'))
        insulin = float(request.form.get('insulin'))
        bmi = float(request.form.get('bmi'))
        diabetesPedigreeFunction=float(request.form.get('diabetesPedigreeFunction'))
        age= int(request.form.get('age'))

        new_data=scaler.transform([[pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,diabetesPedigreeFunction,age]])
        predict = model.predict(new_data)

        if predict[0]==1:
            result="Diabetic"
        else:
            result="Non-Diabetic"
        return render_template('result.html',result=result)
    else:
        return render_template('home.html')
        

if __name__=="__main__":
    app.run(host="0.0.0.0")

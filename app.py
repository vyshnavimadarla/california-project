import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from flask import Flask,request,jsonify,render_template

application=Flask(__name__)
app=application


ridge_model=pickle.load(open('models/lr.pkl', 'rb'))
standard_scaler=pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['Get', 'Post'])
def predict_datapoint():
    if request.method=='POST':
        MedInc=float(request.form.get('MedInc'))
        HouseAge=float(request.form.get('HouseAge'))
        AveRooms=float(request.form.get('AveRooms'))
        AveBedrms=float(request.form.get('AveBedrms'))
        Population=float(request.form.get('Population'))
        AveOccup=float(request.form.get('AveOccup'))
        Latitude=float(request.form.get('Latitude'))
        Longitude=float(request.form.get('Longitude'))

        new_data_sc=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_sc)

        return render_template('index.html', result=result[0])
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
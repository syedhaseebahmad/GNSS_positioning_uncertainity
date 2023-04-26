import pickle
import os
import pandas as pd
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np


app = Flask(__name__)
## Load the model
randomforestmodel = pickle.load(open('GNSS_positioning_uncertainity/randomforestmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods = ['POST'])
def predict_api():
    data_raw = request.json['data']
    print(data_raw)
    data = np.array(list(data_raw.values())).reshape(1,18)
    print(data)
    output = randomforestmodel.predict(data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug= True)


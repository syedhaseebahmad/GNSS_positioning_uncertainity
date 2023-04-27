import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,app,jsonify,url_for,render_template



app = Flask(__name__)
## Load the model
randomforestmodel = pickle.load(open('randomforestmodel.pkl','rb'))

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

@app.route('/predict',methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
   # final_input = StandardScaler.transform(np.array(data).reshape(1,18))
    final_input = np.array(data).reshape(1,18)
    print(final_input)
    output = randomforestmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The predicted uncertainity is {}".format(output) )


if __name__ == "__main__":
    app.run(debug= True)


import pickle
from flask import Flask,request,app,jsonify,render_template,url_for
import numpy as np
import pandas as pd

app = Flask(__name__)
## Load the model
model = pickle.load(open('carmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [x for x in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    print(final_input)
    output = model.predict(final_input)
    return render_template("home.html",prediction_text="The Car Price Prediction is {}".format(round(output[0],2)))


if __name__ == "__main__":
    app.run(debug=True)

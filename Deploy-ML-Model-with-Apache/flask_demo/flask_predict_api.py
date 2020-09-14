#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from flask import Flask, request
# pip install flasgger==0.8.1
from flasgger import Swagger
import numpy as np
import pandas as pd

with open('/var/www/flask_predict_api/rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
  
app = Flask(__name__)
swagger = Swagger(app)


@app.route('/predict', methods=["GET"])
def predict_iris():
    """Example endpoint returning a prediction of iris
    ---
    parameters:
      - name: s_length
        in: query
        type: number
        required: true
      - name: s_width
        in: query
        type: number
        required: true
      - name: p_length
        in: query
        type: number
        required: true
      - name: p_width
        in: query
        type: number
        required: true
    """
    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")

    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
    return str(prediction)


@app.route('/predict_file', methods=["POST"])
def predict_iris_file():
    """Example file endpoint returning a prediction of iris
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    input_data = pd.read_csv(request.files.get("input_file"), header=None)
    prediction = model.predict(input_data)
    return str(list(prediction))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

    # flasgger -> swagger input

# http://127.0.0.1:5000/predict?s_length=5.7&s_width=5.6&p_length=4.3&p_width=7.8

# http://127.0.0.1:5000/apidocs/

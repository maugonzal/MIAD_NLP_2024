#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict_proba(url):

    reg = joblib.load(os.path.dirname(__file__) + '/phishing_clf.pkl')

    url_ = pd.DataFrame([tmp_.split(';')], columns=['mileage', 'state', 'make', 'model', 'antiguedad'])

    p1 = reg.predict(url_)[0,1]

    return p1


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print('Por favor adicione la información del vehículo separada por ";"')

    else:

        url = sys.argv[1]

        p1 = predict_proba(url)

        print(url)
        print('Estimación del precio: ', p1)
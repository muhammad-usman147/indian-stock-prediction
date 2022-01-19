
import numpy as np 
import pandas as pd 
import pickle
from flask import Flask , request, jsonify
'''
parameters required:
open 
high
low
close
VWAP
Volume
Date_from
Date_to

note:date range should be 30
and every data should contain 30 values
sample input
{"Open_mean_lag3": 1129.13330078125,
 "Open_mean_lag7": 1114.1785888671875,
 "Open_mean_lag30": 1121.5511474609375,
 "Open_stddev_lag3": 1.8166667222976685,
 "Open_stddev_lag7": 18.6796817779541,
 "Open_stddev_lag30": 23.634450912475586,
 "High_mean_lag3": 1134.316650390625,
 "High_mean_lag7": 1123.82861328125,
 "High_mean_lag30": 1130.5233154296875,
 "High_stddev_lag3": 0.46666663885116577,
 "High_stddev_lag7": 15.983601570129395,
 "High_stddev_lag30": 19.796167373657227,
 "Low_mean_lag3": 1117.86669921875,
 "Low_mean_lag7": 1102.9356689453125,
 "Low_mean_lag30": 1106.1383056640625,
 "Low_stddev_lag3": 1.6666666269302368,
 "Low_stddev_lag7": 22.8015079498291,
 "Low_stddev_lag30": 23.393524169921875,
 "Close_mean_lag3": 1122.683349609375,
 "Close_mean_lag7": 1115.1285400390625,
 "Close_mean_lag30": 1119.3782958984375,
 "Close_stddev_lag3": 1.433333396911621,
 "Close_stddev_lag7": 13.217548370361328,
 "Close_stddev_lag30": 20.290857315063477,
 "VWAP_mean_lag3": 1124.7900390625,
 "VWAP_mean_lag7": 1113.3021240234375,
 "VWAP_mean_lag30": 1118.00634765625,
 "VWAP_stddev_lag3": 2.5,
 "VWAP_stddev_lag7": 19.93568229675293,
 "VWAP_stddev_lag30": 21.665630340576172,
 "Volume_mean_lag3": 7370550.0,
 "Volume_mean_lag7": 7877574.5,
 "Volume_mean_lag30": 7396477.0,
 "Volume_stddev_lag3": 147733.0,
 "Volume_stddev_lag7": 2133476.75,
 "Volume_stddev_lag30": 2233998.25,
 "month": 1.0,
 "week": 1.0,
 "day": 1.0,
 "day_of_week": 1.0}
'''
app = Flask(__name__)
with open('model.pkl','rb') as f:
    model = pickle.load(f)
@app.route('/detect',methods = ['POST'])
def detect():
    content = request.json
    df = pd.Series({col:value for col,value in content.items()})
    answer = model.predict(n_periods = 1, 
    exogenous= [df])
    

    return jsonify({"Forecasted":str(answer)})

if __name__ == '__main__':
    app.run(debug=True)

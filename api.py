import re
import numpy as np 
import pandas as pd 
import pickle
from tensorflow.keras.models import load_model
import nltk
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import SnowballStemmer
import re
import pickle
from keras.preprocessing.sequence import pad_sequences
nlpmodel = load_model('nlpmodel.h5')

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask , request, jsonify, render_template
'''


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

@app.route('/')
def Home():
    content = {"Open_mean_lag3": 1129.13330078125,
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

    #content = request.json
    df = pd.Series({col:value for col,value in content.items()})
    answer = model.predict(n_periods = 1, 
    exogenous= [df]) 
    
    print("--"*10)
    print("--"*10)
    print(answer)
    with open("answer.txt",'w') as f:
        f.write(str(answer[0]))
    return render_template('index.html', value = str(answer[0]),tweet = '')

text_cleaner = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english") 
def preprocess_text(text, stem=False):

    text = re.sub(text_cleaner, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

@app.route("/tweet",methods = ['POST'])
def sentiment():
    if request.method == "POST":

        text = request.form.get("tweet")
        #text = 'the market seems very awesome to me'
        data = pd.DataFrame({'text' :[text]})
        data.text = data.text.apply(lambda x: preprocess_text(x))
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        tokenizer.fit_on_texts(data.text)
        vocab_size = len(tokenizer.word_index) + 1
        print("Total words", vocab_size)
        length_sequence = 300
        x = pad_sequences(tokenizer.texts_to_sequences(data.text), maxlen=length_sequence)
        pred = nlpmodel.predict(x)
        with open("answer.txt",'r') as f:
            answer = f.readlines()
        if pred[0][0] <= 4: 
            O = 'Negative' 
        else:
            O = "POSITIVE"
        return render_template('index.html',value = str(answer[0]),tweet = str(O))




if __name__ == '__main__':
    app.run(debug=True)

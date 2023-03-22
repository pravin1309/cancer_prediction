import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('cancer.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    #output = round(prediction[0], 2)
    
    if prediction[0]==1:
        return "<h1 style='color:green'>The type of tumour is Benign</h1>"
    else:
        return "<h1 style='color:red'>The type of tumour is Malignant</h1>"
    #return render_template('cancer.html', prediction_text='The type of tumour is {}'.format(output))



app.run(host="0.0.0.0", port=8080)
    


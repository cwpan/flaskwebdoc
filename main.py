import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML page
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output1 = round(prediction[0], 0)
    output = int(output1)
    return render_template('index.html', prediction_text='The coal consumption is {} thousand metric tons.'.format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

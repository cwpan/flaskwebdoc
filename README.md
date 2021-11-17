# Python Flask Energy Consumption Predictor
In this project, we are going to use a linear regression algorithm  from scikit-learn library to help predict the brown coal consumption based on the historical UN energy statistics dataset. Flask light web framework will be use to diliver the analytic portal to the  public.
The dataset is from UN energy statistics dataset

# Overview
This project will be running as MVP to deliver minimum features by leverageing Machine learning model. The data process flow has no different with large scale of marchine learning by follow the processes below
- Data Exporation phase: Power BI Desktop edition was used for data exploration and data clearning. 
- Data Model Creation: Jupytor note book running on Azure compute node or Google Clab notebook was used
- Model deployment: AWS app runner 

# Architecture
![image](https://user-images.githubusercontent.com/11746291/141704112-e44b9c79-b9de-4352-b23f-f71b73dc3ae3.png)

# Model
`model.py` trains and saves the model to disk.
`model.pkl` is the model in pickle format.

```python
# Simple Linear Regression

'''
This model predicts the brown coal consumption in USA by using simple linear regression model.
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json

# Importing the dataset
dataset = pd.read_csv('usa_brown_coal_simplified_all.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set

## linear model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Saving model using pickle
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))
print(model.predict([[1.8]]))

```
# App 
`main.py` has the main function and contains all the required functions for the flask app. Port 8081 and have set debug=True to enable debugging when necessary.
```python
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
    app.run(host='0.0.0.0', port=8081, debug=True)
```

# CI/CD - Deployment via AWS App Runner

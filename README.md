# Python Flask Energy Consumption Predictor
In this project, we are going to use a linear regression algorithm  from scikit-learn library to help predict the brown coal consumption based on the historical UN energy statistics dataset. Flask light web framework will be use to deliver the analytic portal to the  public.
The dataset is from UN energy statistics dataset.

[The Project Flaskweb](https://pg3gixvkxm.us-east-2.awsapprunner.com/predict)

# Acknowledgements
This dataset was kindly published by the United Nations Statistics Division on the UNData site. You can find the original [dataset here](http://data.un.org/Explorer.aspx).

# License
Per the UNData terms of use: all data and metadata provided on UNdata’s website are available free of charge and may be copied freely, duplicated and further distributed provided that UNdata is cited as the reference.

# Overview
This project will be running as MVP to deliver minimum features by leverageing Machine learning model. The data process flow has no different with large scale of marchine learning by follow the processes below
- Data Exporation phase: Power BI Desktop edition was used for data exploration and data clearning. 
- Data Model Creation: Jupytor note book running on Azure compute node or Google Clab notebook was used
- Model deployment: AWS app runner 

# Architecture
![image](https://user-images.githubusercontent.com/11746291/141704112-e44b9c79-b9de-4352-b23f-f71b73dc3ae3.png)

# Data Understanding
By leveraging the Power BI desktop edition, we can even conduct forecast based on the dataset as shown below
![image](https://user-images.githubusercontent.com/11746291/143153365-9b3feeeb-c4b1-44d7-b115-a695896ea238.png)

# Data Preparation
Either use the Microsoft ML notebook `energyall.ipynb` or the Google Colab `AllEnergyAnalysis.ipynb` to create US brown coal consumption from the dataset as 
the input to train the Model.

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
AWS App Runner is an AWS service that provides a fast, simple, and cost-effective way to turn an existing container image or source code directly into a running web service in the AWS Cloud. 

1. Please follow [AWS document links](https://docs.aws.amazon.com/apprunner/latest/dg/getting-started.html) to complete your GitHub repo configuration if you have not done so. 
2. Configure deployment
![image](https://user-images.githubusercontent.com/11746291/142134798-1aa40a45-6168-4782-a376-548f240e8f4c.png)

![image](https://user-images.githubusercontent.com/11746291/142134941-3c182b5b-dffa-4c99-9489-54a588e5b607.png)

![image](https://user-images.githubusercontent.com/11746291/142135042-c852bbc5-d4df-4bc3-8ae7-c0ad5bb4169c.png)

![image](https://user-images.githubusercontent.com/11746291/142135194-c2f21049-7e1d-4c9e-a5af-556db8dd84dd.png)

3. After Service is created successfully, we will able to see the service and its event logs in the service console.
![image](https://user-images.githubusercontent.com/11746291/142135558-5f412b1f-235e-4d7f-8fbd-aff466c54287.png)

4. Any code changes committed to the main branch of the GitHub repository will trigger the CI/CD pipeline to refresh the Flask web as a result.




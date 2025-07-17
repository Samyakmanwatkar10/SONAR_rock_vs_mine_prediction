# Importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Data Preprocessing
# loading the dataset to a pandas DataFrame
sonar_data=pd.read_csv('sonar_data.csv', header=None)

# print(sonar_data.head())
# print(sonar_data.shape)   #(208, 61)
# print(sonar_data.describe())
# print(sonar_data[60].value_counts())   #M 111, R 97
# print(sonar_data.groupby(60).mean())

# separting data and labels
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]
# print(X.shape, Y.shape)  # (208, 60) (208,)

# Training and Testing Data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
# print(X.shape,X_train.shape,X_test.shape)  # (208, 60) (187, 60) (21, 60)
# print(X_train)
# print(Y_train)
# print(X_test)
# print(Y_test)  
# print(Y_test.value_counts())  # M 11, R 10

# model training - logistic regression
model=LogisticRegression()

# training the model
model.fit(X_train,Y_train)

# model evaluation
# accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy on training data:', training_data_accuracy)

X_test_prediction=model.predict(X_test)
test_data_accuaracy=accuracy_score(X_test_prediction,Y_test)
print('accuracy on test data:', test_data_accuaracy)

# making a predictive system
input_data=(0.0217,0.0152,0.0346,0.0492,0.0484,0.0526,0.0773,0.0862,0.1451,0.2110,0.2343,0.2087,0.1645,0.1689,0.1650,0.1967,0.2934,0.3709,0.4309,0.4161,0.5116,0.6501,0.7717,0.8491,0.9104,0.8912,0.8189,0.6779,0.5368,0.5207,0.5651,0.5749,0.5250,0.4255,0.3330,0.2331,0.1451,0.1648,0.2694,0.3730,0.4467,0.4133,0.3743,0.3021,0.2069,0.1790,0.1689,0.1341,0.0769,0.0222,0.0205,0.0123,0.0067,0.0011,0.0026,0.0049,0.0029,0.0022,0.0022,0.0032)
# changing the input_data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)
# reshape the numpy array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)

if prediction[0]=='R':
    print('The object is a Rock')
else:
    print('The object is a Mine')
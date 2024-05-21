# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 11:20:28 2021

@author: okokp
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv('heart.csv')

heart_data.head()

heart_data.tail()

heart_data.shape

heart_data.info()

heart_data.isnull().sum()

heart_data.describe()
print("The Target Value")

print("",heart_data['target'].value_counts())

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

model = KNeighborsClassifier()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

input_data = (52,1,2,258,199,1,1,162,0,0.5,2,0,7)

input_data_as_numpy_array= np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]== 0):
    print('The Person has Heart failure')
elif (prediction[0] == 1):
    print('The Person has Myocardial infraction')
elif (prediction[0] == 2):
    print('The Person has Dilated cardiomyopathy')
elif (prediction[0] == 3):
    print('The Person has Coronary vasaspasm')
elif (prediction[0] == 4):
    print('The Person has Atrial fibrillation')
else:
    
    print('The Person has Arrhythmia- abnormal heart rythmn')
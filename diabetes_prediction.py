import numpy as np
import pandas as pd
#used to standardize data to a common range
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data Collection: Data set from kaggle.com 
df = pd.read_csv('diabetes.csv')

#print(df.groupby('Outcome').mean())

#axis = 1 for column and axis = 0 for row
X = df.drop(columns = 'Outcome', axis=1)
Y = df['Outcome']

#Data Standardization 
scaler = StandardScaler()
scaler.fit(X)
standardized_data= scaler.transform(X)
#or you can do: scaler.fittransform(X) 

#Using X & Y to train our model, X represents data and Y represents model
X = standardized_data
Y = df['Outcome']

#Split dataset into training data and testing data
#test_size = 0.2 means 20% of test data & if we don't include stratify = Y then data might 
#not be split evenly among patients w/ diabetes and patients w/o diabetes
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

#Training the model
classifier = svm.SVC(kernel= 'linear')
#fitting training data into classifier
classifier.fit(x_train, y_train)

#Model Evaluation
x_train_prediction = classifier.predict(x_train)
#finding accuracy score of the model
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
#above 75 is good
#Our Accuracy score: 0.7866449511400652

#Finding accuracy score on test data
x_test_prediction = classifier.predict(x_test)
#finding accuracy score of the model
testing_data_accuracy = accuracy_score(x_test_prediction, y_test)
#Our Accuracy score: 0.7727272727272727
#This model is not overtraining/overfitting so it performs well on both training and testing datasets

#Predictive system
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
#change input data to numpyarray
input_array = np.asarray(input_data)

#reshaping the array because we are only predicting 1 instance
input_reshape = input_array.reshape(1, -1)

#since we standardized data, we cannot send raw information 
std_input = scaler.transform(input_reshape)

prediction = classifier.predict(std_input)
if prediction[0] == 0:
    print("This patient does not have diabetes")
else:
    print("This patient has diabetes")
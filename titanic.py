import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset
titanic = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


d = {'male': 1, 'female': 0}
titanic['Sex'] = titanic['Sex'].map(d)
test['Sex'] = test['Sex'].map(d)
e = {"S":0 , "C":1 , "Q":2}
titanic['Embarked'] = titanic["Embarked"].map(e)
test['Embarked'] = test['Embarked'].map(e)


titanic = titanic.drop([ 'Name' , 'Ticket'] , axis = 1)
test = test.drop(['Name' , 'Ticket'] , axis = 1)


titanic.drop("Cabin",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)


xtrain = titanic.drop(["Survived" ] , axis=1)
ytrain = titanic["Survived"]
xtest = test

xtrain = np.array(xtrain)
xtest = np.array(xtest)
ytrain = np.array(ytrain)



from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(xtrain[:,:])
xtrain[:, :] = imputer.transform(xtrain[:, :])

imputer = imputer.fit(xtest[:,:])
xtest[:,:] = imputer.transform(xtest[:, :])




# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 8))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))




# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
classifier.fit(xtrain, ytrain, batch_size = 15, nb_epoch = 200)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(xtest)
y_pred = y_pred.round()


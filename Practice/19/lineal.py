import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def readData(nameFile):
    auxMatrix = list()  #Matrix of numpy arrays
    with open(nameFile, newline = '') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  #List of headers
    with open(nameFile, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            auxList = list()
            auxList.append(1)   #Adding 1 for convenience
            for i in range(3, len(headers)):
                intAux = float(row[headers[i]])
                auxList.append(intAux)
            auxNum = np.array(auxList)  #Numpy array
            auxNum = np.absolute(auxNum)
            auxMatrix.append(auxNum)
    matrix = np.array(auxMatrix)  #Numpy matrix
    return matrix

def getY(nameFile):
    with open(nameFile, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        auxList = list()
        for row in reader:
            intAux = float(row["price"])
            auxList.append(intAux)
        npArray = np.array(auxList)
    return npArray

##################################################
#                   MAIN
##################################################
# Read data
nameFile = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/19/in.csv'
matrix_X = readData(nameFile)
matrix_y = getY(nameFile)

# Split the data into training/testing sets
matrix_X_train = matrix_X[:-20]
matrix_X_test = matrix_X[-20:]
matrix_y_train = matrix_y[:-20]
matrix_y_test = matrix_y[-20:]

regr = linear_model.LinearRegression() #Linear regression object
regr.fit(matrix_X_train, matrix_y_train) #Training the model
matrix_y_prediction = regr.predict(matrix_X_test) #Making predctions
print('Mean squared error: %.2f' % mean_squared_error(matrix_y_test, matrix_y_prediction))
print('Coefficient of determination: %.2f' % r2_score(matrix_y_test, matrix_y_prediction))
print(" ");
print('########## Values ##########')
for i in range(0, len(matrix_y_test)):
    print("Real: %.2f" % matrix_y_test[i], "Prediction: %.2f" % matrix_y_prediction[i])
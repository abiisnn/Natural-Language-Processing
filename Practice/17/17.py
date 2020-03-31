import csv
import numpy as np

def readData(nameFile):
    auxMatrix = list()  #Matrix of numpy arrays
    with open(nameFile, newline = '') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  #List of headers
        # rowCount = sum(1 for row in csvfile)  # fileObject is your csv.reader
        # print("headers size", len(headers))
        # print(type(headers))
    # print("number of rows: ", rowCount)
    with open(nameFile, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            auxList = list()
            auxList.append(1)   #Adding 1 for convenience
            for i in range(3, len(headers)):
                auxList.append(row[headers[i]])
            auxNum = np.array(auxList).astype(np.float)  #Numpy array
            auxNum = np.absolute(auxNum)
            auxMatrix.append(auxNum)

    matrix = np.array(auxMatrix)  #Numpy matrix
    return matrix

def getY(nameFile):
    with open(nameFile, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        auxList = list()
        for row in reader:
            auxList.append(row["price"])
        auxNum = np.array(auxList).astype(np.float)  #Numpy array

    #Feature Scaling
    U = auxNum.mean()
    S = auxNum.std()
    print("U: ", U)
    print("S: ", S)
    for i in range(0, len(auxNum)):
        auxNum[i] = auxNum[i] - U
        auxNum[i] = abs(auxNum[i])
        if S != 0:
            auxNum[i] = auxNum[i] / S
    return auxNum

def featureScaling(matrix):
    U = np.sum(matrix, axis = 0) #Sum of columns
    for i in range(0, len(U)):
        U[i] = U[i] / len(U)
    
    S = list() #Get Standar D.
    for i in range(0, len(matrix[0])):
        column = [row[i] for row in matrix]
        column = np.array(column)
        std = column.std()
        S.append(std)
    S = np.array(S)
    
    newMatrixAux = list()
    for i in range(0, len(matrix)): #row
        aux = list()
        for j in range(0, len(matrix[i])): #column
            res = matrix[i][j] - U[j]
            res = abs(res)
            if S[j] != 0: 
                res = res / S[j]
            aux.append(res)
        auxNum = np.array(aux)
        newMatrixAux.append(auxNum)
    
    newMatrix = np.array(newMatrixAux)
    return newMatrix

def initializeTheta(n):
    a = list()
    for i in range(0, n):
        a.append(0)
    theta = np.array(a)
    return theta

def printMatrix(matrix, n):
    for i in range(0, n):
        print(matrix[i])


##################################################
#                   MAIN
##################################################
# Read data
nameFile = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/17/in.csv'
matrix = readData(nameFile)
Y = getY(nameFile)
print("Y:")
print(Y)
# print("ORIGINAL:")
# printMatrix(matrix, 5)
#Feature scaling
# print("Feature Scaling")
matrix = featureScaling(matrix)
# printMatrix(matrix, 5)

#Get transpose matrix of data
matrix = matrix.transpose()
# print(len(matrix))
# printMatrix(matrix, 5)

#Get initial theta vector
thetaT= initializeTheta(len(matrix))

#H(theta) = thetaT * matrix
H_theta = thetaT.dot(matrix)
# printMatrix(H_theta, len(H_theta))

#Get Cost Function

sum = 0
for i in range(len(Y)):
    aux = (H_theta[i] - Y[i])
    aux = aux * aux
    sum = sum + aux
sum = sum / (2 * len(Y))

print("PRECIO:", sum)

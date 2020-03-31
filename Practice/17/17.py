import csv
import numpy as np

def readData(nameFile):
    auxMatrix = list()  #Matrix of numpy arrays
    with open(nameFile, newline = '') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  #List of headers
    with open(nameFile, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        cont = 0
        porcent = 15118
        for row in reader:
            if cont < porcent:
                auxList = list()
                auxList.append(1)   #Adding 1 for convenience
                for i in range(3, len(headers)):
                    intAux = float(row[headers[i]])
                    auxList.append(intAux)
                auxNum = np.array(auxList)  #Numpy array
                auxNum = np.absolute(auxNum)
                auxMatrix.append(auxNum)
            else:
                break
            cont = cont + 1

    matrix = np.array(auxMatrix)  #Numpy matrix
    return matrix

def readTest(nameFile):
    auxMatrix = list()  #Matrix of numpy arrays
    with open(nameFile, newline = '') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  #List of headers
    with open(nameFile, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        cont = 0
        porcent = 15118
        for row in reader:
            if cont > porcent:
                auxList = list()
                auxList.append(1)   #Adding 1 for convenience
                for i in range(3, len(headers)):
                    intAux = float(row[headers[i]])
                    auxList.append(intAux)
                auxNum = np.array(auxList)  #Numpy array
                auxNum = np.absolute(auxNum)
                auxMatrix.append(auxNum)
            else:
                cont = cont + 1

    matrix = np.array(auxMatrix)  #Numpy matrix
    return matrix

def getY(nameFile):
    with open(nameFile, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        auxList = list()
        cont = 0
        porcent = 15118
        for row in reader:
            if cont < porcent:
                listaux = list()
                intAux = float(row["price"])
                listaux.append(intAux)
                n = np.array(listaux)
                auxList.append(n)
            else: 
                break
            cont = cont + 1
        npArray = np.array(auxList)
    return npArray

def getYTest(nameFile):
    with open(nameFile, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        auxList = list()
        cont = 0
        porcent = 15118
        for row in reader:
            if cont > porcent:
                listaux = list()
                intAux = float(row["price"])
                listaux.append(intAux)
                n = np.array(listaux)
                auxList.append(n)
            else: 
                cont = cont + 1
        npArray = np.array(auxList)
    return npArray
    
def featureScaling(matrix):
    U = np.sum(matrix, axis = 0) #Sum of columns

    for i in range(0, len(U)):
        U[i] = U[i] / len(matrix)
    
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
            if j == 0:
                res = 1
            aux.append(res)
        auxNum = np.array(aux)
        newMatrixAux.append(auxNum)
    
    newMatrix = np.array(newMatrixAux)
    return newMatrix

def initializeTheta(n):
    m = list()
    for i in range(0, n):
        aux = list()
        aux.append(0)
        auxNum = np.array(aux)
        m.append(auxNum)

    theta = np.array(m)
    return theta

def printMatrix(matrix, n):
    for i in range(0, n):
        print(matrix[i])

def sumaCostFunction(H_theta, Y):
    auxSub = H_theta - Y
    ansSum = 0
    for i in auxSub:
        for j in i:
            j = j * j
            ansSum = ansSum + j
    ansSum = ansSum / (2 * len(Y[0]))
    return ansSum

def gradientDescent(theta, H_theta, Y, matrix, learningRate):
    temp = list()
    for j in range(0, len(matrix)):
        ans = 0
        for i in range(0, len(matrix[j])):
            res = ((H_theta[0][i] - Y[0][i])) * matrix[j][i]
            ans = ans + res
        ans = ans / len(H_theta[0])
        aux =  theta[j][0] - (learningRate * ans)
        listAux = list()
        listAux.append(aux)
        temp.append(listAux)

    tempNum = np.array(temp)
    return tempNum

##################################################
#                   MAIN
##################################################
# Read data
nameFile = '/Users/abiga/Desktop/AbiiSnn/GitHub/Natural-Language-Processing/Practice/17/in.csv'
matrixN = readData(nameFile) #Training set
auxY = getY(nameFile)
Y = auxY.transpose()

matrixTestN = readTest(nameFile) #Testing set

# printMatrix(matrixN, 5)
# printMatrix(matrixTestN, 5)

auxY = getYTest(nameFile) 
YTest = auxY.transpose()


#Feature scaling
matrixN = featureScaling(matrixN)
matrixTestN = featureScaling(matrixTestN)

#Get transpose matrix of data
matrix = matrixN.transpose()
matrixTest = matrixTestN.transpose()

#Get initial theta vector
theta = np.zeros(shape = (len(matrix), 1)) # n x 1

#H(theta) = thetaT * matrix
thetaT = theta.transpose() # 1 x n
H_theta = thetaT.dot(matrix) # 1 x m

#Get Cost Function
price = sumaCostFunction(H_theta, Y) 
print("Cost Function initial value:", price)

tempTheta = initializeTheta(len(matrix))
learningRate = 0.1

print("TRAINING TEST:")
for ite in range(0, 1000):
    tempTheta = gradientDescent(theta, H_theta, Y, matrix, learningRate)
    theta = tempTheta
    thetaT = tempTheta.transpose()
    H_theta = thetaT.dot(matrix)
    price = sumaCostFunction(H_theta, Y)
    if((ite % 50) == 0):
        print("Iteration:", ite, "Cost Function value:", price)


#H(theta) = thetaT * matrix
thetaT = theta.transpose() # 1 x n
H_theta = thetaT.dot(matrixTest) # 1 x m

#Get Cost Function
price = sumaCostFunction(H_theta, YTest) 
print("Cost Function initial value:", price)
# tempTheta = initializeTheta(len(matrixTest))
learningRate = 0.1
print("")
print("TESTING SET:")
for ite in range(0, 1000):
    tempTheta = gradientDescent(theta, H_theta, YTest, matrixTest, learningRate)
    theta = tempTheta
    thetaT = tempTheta.transpose()
    H_theta = thetaT.dot(matrixTest)
    price = sumaCostFunction(H_theta, YTest)
    if((ite % 50) == 0):
        print("Iteration:", ite, "Cost Function value:", price)


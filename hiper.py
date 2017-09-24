import numpy as np
from numpy import linalg as li

# genRandomHV(): generates random hypervector
# D: number of dimensions
def genRandomHV(D):
    if (D % 2) == 1:
        print("Dimension is odd.")
    else:
        hv = np.random.permutation(D)
        for x in range(D):
            if hv[x] < D/2:
                hv[x] = -1
            else:
                hv[x] = 1
        return hv

# cosAngle(): gives cosine angle between two hypervectors
# x, y: input hypervectors
def cosAngle(x, y):
    return (np.dot(x,y) / (li.norm(x) * li.norm(y)))

# lookupItemMemory(): takes in inputted item memory, checks if given key 
#   gas an associated hypervector, if it does not, will generate a new
#   hypervector for it and return newly appended memory
# inputMemory: associated memory that contains keys and hypervectors,
#   is a dictionary type
# item: key which is used to look into inputMemory and pull out associated
#   hypervector
# D: number of dimensions used for hypervector 
def lookupItemMemory(inputMemory, item, D):
    if item in inputMemory.keys():
        return inputMemory, inputMemory[item]
    else:
        inputMemory[item] = genRandomHV(D)
        return inputMemory, inputMemory[item]

# binarizeHV(): returns binarized hypervector
# inputHV: input hypervector
# "Hypervector is binarized about threshold
def binarizeHV(inputHV):
    threshold = 0
    for i in range(len(inputHV)):
        if inputHV[i] > threshold:
            inputHV[i] = 1
        else:
            inputHV[i] = -1
    return inputHV

# N is the n-grams value
# inputBuffer is string value
def nGramHV(inputBuffer, itemMemory, N, D):
    block = np.zeros((N, D), dtype=np.int)
    sumHV = np.zeros((1,D), dtype=np.int)

    for i in range(len(inputBuffer)):
       bufferKey = inputBuffer[i]
       block = np.roll(block, 1, axis = 0)
       block = np.roll(block, 1, axis = 1)
       
       itemMemory, block[0] = lookupItemMemory(itemMemory, bufferKey, D)

       if i >= N:
           nGrams = block[0]
           for j in range(1,N):
               nGrams = nGrams * block[j]
           sumHV = sumHV + nGrams
    return itemMemory, sumHV[0]

# needs work
# This function eats pandas and returns a list
# Maybe should return a list of tags as well
def buildnGramModel(inputBuffers, inputMemory, N, D):
    #hdcModel =  {} #Was previously a list
    numOfBuffers = len(inputBuffers)

    hdcModel = [np.zeros(N)] * numOfBuffers
    #maybe change to range(len(inputDataFrame[bufferClass]))

    # % indicator not working too well, double check it 
    timer = 0
    print("Processing", numOfBuffers, "element(s).")

    barL = numOfBuffers
    for i in range(numOfBuffers):
        inputMemory, hdcModel[i] = nGramHV(inputBuffers[i], inputMemory, N, D)

        printProgressBar(i + 1, barL, prefix = 'Progress:', suffix = 'Complete', length = 30)
 
        #if i % (np.ceil(numOfBuffers / 20)) == 0:
        #    print(timer, "%")
        #    timer += 5
    
    #print("Done")

    return inputMemory, hdcModel

# This function will take in indexes and return an ID hypervector
def getID(index, inputMemory, D):
    #identifier = 'id_x' +x' + str(xIndex) + 'y' + str(yIndex)

    # zfill() is used for leading zeros
    identifier = 'id_' + str(index).zfill(7)
    inputMemory, returnHV = lookupItemMemory(inputMemory, identifier, D)
    return inputMemory, returnHV

# Written from memory, maybe check against matlab code
# Havent checked if this is working
def idHV(inputBuffer, itemMemory, D):
    returnHV = np.zeros(D)

    itemMemory, HV = lookupItemMemory(itemMemory, inputBuffer, D)
    return itemMemory, HV

# Maybe make a get ID helper function for this
# N is unnecessary, should be removed as an input argument
# TODO: remove dataframe
def buildIDModel(inputBuffers, inputMemory, D):

    numOfBuffers = len(inputBuffers)

    hdcModel = [np.zeros(D)] * numOfBuffers

    #timer = 0
    print("Processing", numOfBuffers, "element(s).")

    barL = numOfBuffers
    for i in range(numOfBuffers):
        inputMemory, hdcModel[i] = idHV(inputBuffers[i], inputMemory, D)

        #printProgressBar(i + 1, barL, prefix = 'Progress:', suffix = 'Complete', length = 30)

        #if i % (np.ceil(numOfBuffers / 20)) == 0:
        #    print(timer, "%")
        #    timer += 5
    
    #print("Done")

    return inputMemory, hdcModel

# Eats lists and returns a one hv per class
def oneHvPerClass(inputLabels, inputHVs, D):
    #inputLabels and inputHV are lists
    
    #This creates a list with no duplicates
    labels = list(set(inputLabels))

    #initializing output labels list and hv list
    outLabels = [" "] * len(labels)
    outHV = [np.zeros(D)] * len(labels)
    
    for i in range(len(labels)):
        
        currentLabel = labels[i]
        indices = [i for i, x in enumerate(inputLabels) if x == currentLabel]
        
        #outputHV initialized
        theSingleHVtbc = np.zeros(D)

        for index in indices:
            theSingleHVtbc = theSingleHVtbc + inputHVs[index]
        
        outLabels[i] = currentLabel
        outHV[i] = theSingleHVtbc
    
    return outLabels, outHV

# This function attempts to guess the class of the input vector based on the model given
# The given model should be one HV per class
def checkVector(trainedVectorLabels, trainedVectorModel, inputHV):

    returnClass = trainedVectorModel[0]
    largestAngle = 0
    
    for i in range(len(trainedVectorModel)):
        currentCosAngle = cosAngle(inputHV, trainedVectorModel[i])
        if currentCosAngle > largestAngle:
            #print(currentCosAngle,"is larger than", largestAngle)
            returnClass = trainedVectorLabels[i]
            largestAngle = currentCosAngle
            
    return returnClass

# This function runs an accuracy check based on the inputVectorModel and
#  train vectors given
# The given model should be one HV per class
# Use oneHvPerClass to achieve this
# TODO: Confusion Matrix
def accuracyCheck(testVectorLabel, testVectorModel, trainVectorLabel, trainVectorModel):
    attempts = 0
    correct = 0
    classAttempts = {}
    classCorrect = {}
    classAccuracy = {}

    for label in trainVectorLabel:
        classAttempts[label] = 0
        classCorrect[label] = 0
    
    #print('Going to', range(len(trainVectorModel)))
    for i in range(len(testVectorModel)):
        guess = checkVector(trainVectorLabel, trainVectorModel, testVectorModel[i])
        attempts = attempts + 1
        classAttempts[testVectorLabel[i]] = classAttempts[testVectorLabel[i]] + 1
        #print(i)
        #print(guess)
        #print(testVectorLabel[i])
        if testVectorLabel[i] == guess:
            correct = correct + 1
            classCorrect[guess] = classCorrect[guess] + 1 
        #else:
        #    print("incorrect")
    
    accuracy = (correct/attempts) * 100

    for label in classAttempts.keys():
        classAccuracy[label] = (classCorrect[label] / classAttempts[label]) * 100
    
    return accuracy, classAccuracy

class HDCModel(object):
    size = 0
    buffers = []
    classes = []
    
    dimensions = 0
    
    idHvs = []
    nGramHvs = {}
    
    training = True
    
    oneClass = []
    oneidHvs = []
    onenGramHvs = {}
    
    
    # buffers 
    def __init__(self, buffers, classes, training, dimensions):
        self.buffers = buffers
        self.classes = classes
        self.training = training
        self.dimensions = dimensions
        self.size = len(buffers)
        
        if len(buffers) != len(classes):
            print("Buffers and labels are not the same size")
            
    def buildID(self, itemMemory):
        print("Building an ID model")
        itemMemory, self.idHvs = buildIDModel(self.buffers, itemMemory, self.dimensions)
        
        #self.oneClass, self.oneidHvs = oneHvPerClass(self.classes, self.idHvs, self.dimensions)
        
        return itemMemory
        
        
    # Maybe make multiple nGram models per class
    def buildnGram(self, itemMemory, N):
        print("Building an nGram model with N =", N)
        itemMemory, self.nGramHvs[N] = buildnGramModel(self.buffers, itemMemory, N, self.dimensions)
        
        self.oneClass, self.onenGramHvs[N] = oneHvPerClass(self.classes, self.nGramHvs[N], self.dimensions)
        
        return itemMemory
'''
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '.'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + ' ' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
'''

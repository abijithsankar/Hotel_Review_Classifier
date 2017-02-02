import sys,string,os,math


def loadStopWords():
    '''
    This function load the list of stop words and 
    returns the stop word list
    Args:
        None
    Returns:
        stopWordList: list of stop words
    '''
    stopWordPath = "Z:/MSBooks/NLP/HW2/stop-words.txt"
    stopWordList = []
    with open(stopWordPath) as stopWords:
        stopWordList = stopWords.readline().split("|")
    return stopWordList
    
def loadTestData(testTextPath):
    '''
    This function loads the test data and return
    the test data in the form of a list
    Args:
        None
    Returns:
        testDataList: test data in the form of list
    '''
    testDataList=[]
    with open(testTextPath) as testData:
        testDataList = testData.readlines()
    testDataList = [x.strip() for x in testDataList]
    return testDataList

def getNbModel():
    '''
    This function retrieves the model parameters created by
    the learner and returns the total instances per class
    and term probabilities per class
    Args:
        None
    Returns:
        classSums: list of sum of feature instances of each class in training data
        freeqProbs: list of terms and their class probabilities
    '''
        
    modelPath = "Z:/MSBooks/NLP/HW2/nbmodel.txt"
    count = 1
    classSums = []
    freeqProbs = []
    modelParameterList = []
    with open(modelPath) as modelParams:
        modelParameterList = [x.strip("\r\n") for x in modelParams.readlines()]

    for item in modelParameterList:
        if(count == 1):
            classSums = item
            count = 2
        else:
           freeqProbs.append(item)
    return classSums,freeqProbs
    
def getClassProbs(classSums):
    '''
    This function returns the prior probabilities of each class
    Args:
        classSums: list of sum of feature instances of each class in training data
    Returns:
        truthProb: prior probability of truthful class
        deceptiveProb: prior probability of deceptive class
        positiveProb: prior probability of positive class
        negativeProb: prior probability of negative class
    '''
    classSumList = classSums.split("|")
    truthfulSum,deceptiveSum,negativeSum,positiveSum = float(classSumList[0]),float(classSumList[1]),float(classSumList[2]),float(classSumList[3])
    truthProb = truthfulSum/float(truthfulSum + deceptiveSum)
    deceptiveProb = deceptiveSum/float(truthfulSum + deceptiveSum)
    positiveProb = positiveSum/float(negativeSum + positiveSum)
    negativeProb = negativeSum/float(negativeSum + positiveSum)  
    return truthProb,deceptiveProb,positiveProb,negativeProb
    
def getFreeqDict(freeqProbs):
    '''
    This function creates and returns a mapping from features to their
    class conditional probabilities
    Args:
        freeqProbs: list of terms and their class probabilities
    Returns:
        freeqDict: mapping between features and their class conditional probabilities
    '''
    freeqDict = {}
    for item in freeqProbs:
        keyAndValues = item.split("|")
        key = keyAndValues[0]
        values = keyAndValues[1:]
        freeqDict[key] = values
    return freeqDict 
    
def getProcessedTokens(testDataList,stopWordList,freeqDict):
    '''
    This function process the test data features, removes stop words and
    ignores unknown features and returns the final test data list
    Args:
        testDataList: test data in the form of list
        stopWordList: list of stop words
        freeqDict: mapping between features and their class conditional probabilities
    Returns:
        finalTestDataList: processed test data features
    '''
    finalTestDataList = []        
    for item in testDataList:
        itemList = item.split()
        hotelId = itemList[0]
        reviewTermsList = itemList[1:]
        reviewTermsList = [x.lower() for x in reviewTermsList]
        itemList = removeStopWords(reviewTermsList,stopWordList)
        itemList = removeUnknownTokens(itemList,freeqDict)
        itemList.insert(0,hotelId)
        finalTestDataList.append(itemList)
    return finalTestDataList
    
def removeStopWords(testDataList, stopWordList):
    '''
    This function removes stop words from test data
    and returns the processed test data
    Args:
        testDataList: test data in the form of list
        stopWordList: list of stop words
    Returns:
        finalList: processed test data
    '''
    finalList = []
    for w in testDataList:
        if w not in stopWordList:
            word = w.translate(None, string.punctuation)
            finalList.append(word)
            
    finalList = filter(None, finalList)    
    return finalList
    
def removeUnknownTokens(itemList,freeqDict):
    '''
    This function removes unknown tokens in test data
    and returns processed test data
    Args:
        itemList: test data in the form of list
        freeqDict: mapping between features and their class conditional probabilities
    Returns:
        finalList: processed test data
    '''
    finalList = []
    for w in itemList:
        if w in freeqDict:
            finalList.append(w)
    return finalList


    
 

    
def getTestReviewDict(finalTestDataList):
    '''
    This function creates a mapping between review ids and review features
    Args:
        finalTestDataList: processed test data features
    Returns:
        testReviewDict: mapping between review ids and review features
    '''
    testReviewDict = {}
    for item in finalTestDataList:
        testReviewDict[item[0]] = item[1:]
     
    return testReviewDict
     

def predict(testReviewDict,freeqDict,truthProb,deceptiveProb,positiveProb,negativeProb):
    '''
    This function makes the prediction for unknown test data set
    Args:
        testReviewDict: mapping between review ids and review features
        freeqDict: mapping between features and their class conditional probabilities
        truthProb: prior probability of truthful class
        deceptiveProb: prior probability of deceptive class
        positiveProb: prior probability of positive class
        negativeProb: prior probability of negative class
    Returns:
        predictionDict: mapping between review ids and their class predictions
        
    '''
    predictionDict = {}       
    for key,value in testReviewDict.iteritems():       
        p_truthful_review = math.log(truthProb)
        p_deceptive_review = math.log(deceptiveProb)
        p_positive_review = math.log(positiveProb)
        p_negative_review = math.log(negativeProb)
        predictionList = []
        for word in value:
            p_truthful_review = p_truthful_review + math.log(float(freeqDict[word][0]))
            p_deceptive_review = p_deceptive_review + math.log(float(freeqDict[word][1]))
            p_positive_review = p_positive_review + math.log(float(freeqDict[word][2]))
            p_negative_review = p_negative_review + math.log(float(freeqDict[word][3]))
            
        if p_truthful_review > p_deceptive_review:
            predictionList.append("truthful")
        elif p_deceptive_review > p_truthful_review:
            predictionList.append("deceptive")
        if p_positive_review  > p_negative_review:
            predictionList.append("positive")
        elif p_negative_review > p_positive_review:
            predictionList.append("negative")    
        
        predictionDict[key] = predictionList
        
    
    return predictionDict
        

    
def createOutputFile(predictionDict):
    '''
    This function creates the final output file with all the
    predictions on the unknown test data
    Args:
        predictionDict: mapping between review ids and their class predictions
    Returns:
        None
    '''
    nbOutput = open("Z:/MSBooks/NLP/HW2/nboutput.txt","w")
    for key,value in predictionDict.iteritems():
        if(len(value) == 2):
           nbOutput.write("{} {} {}".format(key,value[0],value[1]) + os.linesep) 
        
        
     
                
def main(testTextPath):
    '''
    This is the Naiive Bayes Classifier that make prediction
    on unseen test data set
    '''
    stopWordList = loadStopWords() 
    
    testDataList = loadTestData(testTextPath)
    
    classSums,freeqProbs = getNbModel()
       
    truthProb,deceptiveProb,positiveProb,negativeProb = getClassProbs(classSums)
      
    freeqDict = getFreeqDict(freeqProbs)
    
    finalTestDataList = getProcessedTokens(testDataList,stopWordList,freeqDict)
    
    testReviewDict = getTestReviewDict(finalTestDataList)
    
    predictionDict= predict(testReviewDict,freeqDict,truthProb,deceptiveProb,positiveProb,negativeProb)
   
    createOutputFile(predictionDict)
   
   
   
    
main(sys.argv[1])
import sys
import string
import copy
import os

def loadStopWords():
    '''
    This function loads the stop words file and 
    returns the list of stop words
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
    
def loadTrainData(trainTextPath):
    '''
    This function loads the hotel reviews file for training
    and returns the review file as a list
    Args:
        trainTextPath: path of training data file
    Returns:
        trainDataList: review file in the form of list
    '''
    trainDataList=[]
    with open(trainTextPath) as trainData:
        trainDataList = trainData.readlines()    
    trainDataList = [x.strip() for x in trainDataList]
    
    return trainDataList
    
def getProcessedTokens(trainDataList,stopWordList):
    '''
    This function process the training data, does feature extraction
    by removing stop words and returns the final training data
    in the form of a list
    Args:
        trainDataList: training data in the form of list
        stopWordList: list of stop words
    Returns:
        finalTrainDataList: final processed training data in the form of list
    '''
    finalTrainDataList = []        
    for item in trainDataList:
        itemList = item.split()
        hotelId = itemList[0]
        reviewTermList = itemList[1:]
        reviewTermList = [x.lower() for x in reviewTermList]
        itemList = removeStopWords(reviewTermList,stopWordList)
        itemList.insert(0,hotelId)
        finalTrainDataList.append(itemList)
    return finalTrainDataList

def removeStopWords(trainDataList, stopWordList):
    '''
    This function removes the stopWords from a review text and
    returns the processed review
    Args:
        trainDataList: list of words in the training data in each review
        stopWordList: list of stop words
    Returns:
        finalList: processed review after removing stop words
    '''
    finalList = []
    for w in trainDataList:
        if w not in stopWordList:
            word = w.translate(None, string.punctuation)
            finalList.append(word)
            
    finalList = filter(None, finalList)    
    return finalList
    




def loadLabeledData(trainLabelPath):
    '''
    This function loads the labels of the training data and
    returns the labels as a list
    Args:
        trainLabelPath: path to the label data file
    Returns:
        labelList: labels in the form of list
    '''
    labelList=[]
    with open(trainLabelPath) as labelData:
        labelList = labelData.readlines()       
    labelList = [x.strip() for x in labelList]
    return labelList
    

    
def getProcessedLabels(labelDataList):
    '''
    This function process the labels as required by the feature selection 
    method employed in the model
    Args:
        labelDataList: list of labels
    Returns:
        finalLabelList: processed labels
    '''
    finalLabelList = []
    for item in labelDataList:
        itemList = item.split()
        hotelId = itemList[0]
        classList = itemList[1:]
        classList = [x.lower() for x in classList]
        itemList[0] = hotelId
        finalLabelList.append(itemList)
    return finalLabelList
    
def getDictionaries(finalTrainDataList,finalLabelList):
    '''
    This function creates mappings between review Id and review; 
    review Id and labels and returns these two mappings
    Args:
        finalTrainDataList: processed training data
        finalLablelList: processed labels
    Returns:
        reviewDict: mapping between review id and review text
        labelDict: mapping between review id and label data
    '''
    reviewDict = {}
    for items in finalTrainDataList:
        reviewDict[items[0]] = items[1:]    
    labelDict = {}
    for items in finalLabelList:
        labelDict[items[0]] = items[1:]
    return reviewDict,labelDict
    
def getDataFrame(reviewDict,labelDict):
    '''
    This function creates a dataframe which is a freequency table
    that has term freequencies in each class and 
    returns the dataframe
    Args:
        reviewDict: mapping between review id and review text
        labelDict: mapping between review id and label data
    Returns:
        dataFrame: term freequency table dataframe
    '''
    dataFrame = {}
    wordFlagList = []
    for key,value in reviewDict.iteritems():
        label1 = labelDict[key][0]
        label2 = labelDict[key][1]
        for item in value:
            
            if item not in wordFlagList:
                truthful = 0
                deceptive = 0
                positive = 0
                negative = 0
                if label1 == 'truthful':
                    truthful = 1
                elif label1 == 'deceptive':
                    deceptive = 1
                if label2 == 'positive':
                    positive = 1
                elif label2 == 'negative':
                    negative = 1
                
                dataFrame[item] = {'truthful':truthful,'deceptive':deceptive,'positive':positive,'negative':negative}
                wordFlagList.append(item)
            elif item in wordFlagList:
                if label1 == 'truthful':
                    dataFrame[item]['truthful'] = dataFrame[item]['truthful'] + 1
                elif label1 == 'deceptive':
                    dataFrame[item]['deceptive'] = dataFrame[item]['deceptive'] + 1
                if label2 == 'positive':
                    dataFrame[item]['positive'] = dataFrame[item]['positive'] + 1
                elif label2 == 'negative':
                    dataFrame[item]['negative'] = dataFrame[item]['negative'] + 1
                
        
    return dataFrame
    


    
def createNBModel(dataFrame):
    '''
    This function creates the naiive bayes model by calculating the 
    feature probabilities and returns model and associated 
    class probabilities
    Args:
        dataFrame: term freequency table dataframe
    Returns:
        modelDataFrame: dataframe with feature conditional probabilities
        truthfulSum: total number of feature instances in the 'Truthful' class
        deceptiveSum: total number of feature instances in the 'Deceptive' class
        negativeSum: total number of feature instances in the 'Negative' class
        positiveSum: total number of feature instances in the 'Positive' class
    '''
    vocabSize = len(dataFrame)
    smoothVariable = 1
    deceptiveSum,truthfulSum,negativeSum,positiveSum = 0,0,0,0
    
    for key,value in dataFrame.iteritems():
        deceptiveSum = deceptiveSum + value['deceptive']
        truthfulSum = truthfulSum + value['truthful']
        negativeSum = negativeSum + value['negative']
        positiveSum = positiveSum + value['positive']
        
    modelDataFrame = copy.deepcopy(dataFrame)
    for key,items in dataFrame.iteritems():
        modelDataFrame[key]['truthful'] = (dataFrame[key]['truthful'] + smoothVariable) / float(truthfulSum + vocabSize)
        modelDataFrame[key]['deceptive'] = (dataFrame[key]['deceptive'] + smoothVariable) / float(deceptiveSum + vocabSize)
        modelDataFrame[key]['negative'] = (dataFrame[key]['negative'] + smoothVariable) / float(negativeSum + vocabSize)
        modelDataFrame[key]['positive'] = (dataFrame[key]['positive'] + smoothVariable) / float(positiveSum + vocabSize)
    return modelDataFrame,truthfulSum,deceptiveSum,negativeSum,positiveSum
    
def writeModelFile(modelDataFrame,truthfulSum,deceptiveSum,negativeSum,positiveSum):
    '''
    This function creates a model output file that stores all the model parameters
    Args:
        modelDataFrame: dataframe with feature conditional probabilities
        truthfulSum: total number of feature instances in the 'Truthful' class
        deceptiveSum: total number of feature instances in the 'Deceptive' class
        negativeSum: total number of feature instances in the 'Negative' class
        positiveSum: total number of feature instances in the 'Positive' class
    Returns:
        None
    '''
    nbmodel = open("Z:/MSBooks/NLP/HW2/nbmodel.txt","w")
    nbmodel.write("{}|{}|{}|{}".format(truthfulSum,deceptiveSum,negativeSum,positiveSum) + os.linesep)
    for key,items in modelDataFrame.iteritems():
        line = "{}|{}|{}|{}|{}".format(key,items['truthful'],items['deceptive'],items['positive'],items['negative'])
        nbmodel.write(line + os.linesep)

def main(trainTextPath, trainLabelPath):
    '''
    This is the Naiive Bayes learner for prediciting whether a given review is
    Positive or Negative and Truthful or Deceptive
    '''
    stopWordList = loadStopWords()
    
    trainDataList = loadTrainData(trainTextPath)
    
    finalTrainDataList = getProcessedTokens(trainDataList,stopWordList)
    
    labelDataList = loadLabeledData(trainLabelPath)
    
    finalLabelList = getProcessedLabels(labelDataList)
    
    reviewDict,labelDict = getDictionaries(finalTrainDataList,finalLabelList)
    
    dataFrame = getDataFrame(reviewDict,labelDict)
    
    modelDataFrame,truthfulSum,deceptiveSum,negativeSum,positiveSum = createNBModel(dataFrame)
    
    writeModelFile(modelDataFrame,truthfulSum,deceptiveSum,negativeSum,positiveSum)
    
        
    
    
    
    
            
                
            
                
    
    
    
    
    
    
    
main(sys.argv[1],sys.argv[2])
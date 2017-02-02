import collections
def main():
    originalLabelFile = "Z:/MSBooks/NLP/HW2/train-labels.txt"
    predictedFile = "Z:/MSBooks/NLP/HW2/nboutput.txt"
    originalList = []
    predictionList = []
    with open(originalLabelFile) as original:
        originalList = original.readlines()
        originalList = originalList[:100] + originalList[500:1000]
        originalList = [x.strip("\n") for x in originalList]
    with open(predictedFile) as prediction:
        predictionList = [x.strip("\r\n") for x in prediction.readlines()]
    originalDict = {}
    predictionDict = {}
    for item in originalList:
        keyAndValue = item.split()
        key = keyAndValue[0]
        value = keyAndValue[1:]
        originalDict[key] = value
    for item in predictionList:
        keyAndValue = item.split()
        key = keyAndValue[0]
        value = keyAndValue[1:]
        predictionDict[key] = value
        
    truthfulTP,truthfulFP,truthfulFN = 0,0,0
    deceptiveTP,deceptiveFP,deceptiveFN = 0,0,0
    positiveTP,positiveFP,positiveFN = 0,0,0
    negativeTP,negativeFP,negativeFN = 0,0,0
    
    
    for key,value in predictionDict.iteritems():
        if(value[0] == "truthful" and originalDict[key][0] == "truthful"):
            truthfulTP = truthfulTP + 1
        elif(value[0] == "truthful" and originalDict[key][0] == "deceptive"):
            truthfulFP = truthfulFP + 1
        elif(value[0] =="deceptive" and originalDict[key][0] == "truthful"):
            truthfulFN = truthfulFN + 1
            
        if(value[0] == "deceptive" and originalDict[key][0] == "deceptive"):
            deceptiveTP = deceptiveTP + 1
        elif(value[0] == "deceptive" and originalDict[key][0] == "truthful"):
            deceptiveFP = deceptiveFP + 1
        elif(value[0] =="truthful" and originalDict[key][0] == "deceptive"):
            deceptiveFN = deceptiveFN + 1
            
        if(value[1] == "positive" and originalDict[key][1] == "positive"):
            positiveTP = positiveTP + 1
        elif(value[1] == "positive" and originalDict[key][0] == "negative"):
            positiveFP = positiveFP + 1
        elif(value[1] =="negative" and originalDict[key][1] == "positive"):
            positiveFN = positiveFN + 1
        
        
        if(value[1] == "negative" and originalDict[key][1] == "negative"):
            negativeTP = negativeTP + 1
        elif(value[1] == "negative" and originalDict[key][0] == "positive"):
            negativeFP = negativeFP + 1
        elif(value[1] =="positive" and originalDict[key][1] == "negative"):
            negativeFN = negativeFN + 1
            
    
    truthfulPrecision = truthfulTP/float(truthfulTP + truthfulFP)
    truthfulRecall = truthfulTP/float(truthfulTP + truthfulFN)
    
    deceptivePrecision = deceptiveTP/float(deceptiveTP + deceptiveFP)
    deceptiveRecall = deceptiveTP/float(deceptiveTP + deceptiveFN)
    
    positivePrecision = positiveTP/float(positiveTP + positiveFP)
    positiveRecall = positiveTP/float(positiveTP + positiveFN)
    
    negativePrecision = negativeTP/float(negativeTP + negativeFP)
    negativeRecall = negativeTP/float(negativeTP + negativeFN)
    
    truthfulF1 = (2 * truthfulPrecision * truthfulRecall) / float(truthfulPrecision + truthfulRecall)
    deceptiveF1 = (2 * deceptivePrecision * deceptiveRecall) / float(deceptivePrecision + deceptiveRecall)
    positiveF1 = (2 * positivePrecision * positiveRecall) / float(positivePrecision + positiveRecall)
    negativeF1 = (2 * negativePrecision * negativeRecall) / float(negativePrecision + negativeRecall)
    
    
    finalF1Score = (truthfulF1 + deceptiveF1 + positiveF1 + negativeF1)/4.0
    
    
    
    print "Deceptive",deceptivePrecision,deceptiveRecall,deceptiveF1
    print "Truthful",truthfulPrecision,truthfulRecall,truthfulF1
    print "Positive",positivePrecision,positiveRecall,positiveF1
    print "Negative",negativePrecision,negativeRecall,negativeF1
    print finalF1Score
        
            
    
    
    
        
    
    
   
        
        
        
main()
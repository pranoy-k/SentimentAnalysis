import sys
import getopt
import os
import math
import operator
from collections import Counter
import numpy as np

class Maxent:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Maxent initialization"""
    
    self.numFolds = 10
    

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Maxent classifier 

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """



    # Write code here

    return 'pos'
  
  def sigmoid(x):
    return 1 / ( 1 + np.exp(x))

  def cost_function(feature_vec, klass):
    grad = np.zeros

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Maxent class.
     * Returns nothing
    """

    # Prepare Feature vector
    feature_vec = np.zeros(len(self.bag_of_words))
    for word in words:
      feature_vec[self.word_to_int[word]] = 1
    
    cost,grad = self.cost_function(feature_vec, klass)


  def train(self, split, epsilon, eta, lambdaa):
      """
      * TODO 
      * iterates through data examples
      """
      self.bag_of_words = set()

      for example in split.train:
          words = example.words
          self.bag_of_words.update(words)
      self.bag_of_words = list(self.bag_of_words)
      self.word_to_int = {word:i for i,word in enumerate(self.bag_of_words)}
      self.int_to_word = {i:word for i,word in enumerate(self.bag_of_words)}

      self.weights = np.random.random(len(self.bag_of_words))
      self.bias = np.random.random(1)

      for example in split.train:
          words = example.words
          self.addExample(example.klass, words)
      

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  

def test10Fold(args):
  pt = Maxent()
  
  splits = pt.crossValidationSplits(args[0])
  epsilon = float(args[1])
  eta = float(args[2])
  lambdaa = float(args[3])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Maxent()
    accuracy = 0.0
    classifier.train(split, epsilon, eta, lambdaa)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy
    
    
def classifyDir(trainDir, testDir, eps, et, lamb):
  classifier = Maxent()
  trainSplit = classifier.trainSplit(trainDir)
  epsilon = float(eps)
  eta = float(et)
  lambdaa = float(lamb)
  classifier.train(trainSplit, epsilon, eta, lambdaa)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print '[INFO]\tAccuracy: %f' % accuracy
    
def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  print("len(args)",len(args))
  if len(args) == 5:
    classifyDir(args[0], args[1], args[2], args[3], args[4])
  elif len(args) == 4:
    test10Fold(args)

if __name__ == "__main__":
    main()

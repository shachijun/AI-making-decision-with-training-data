# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math


class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False  # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([f for datum in trainingData for f in datum.keys()]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        "*** YOUR CODE HERE ***"
        # main idea: find label y that maximize P(f1,f2,..,fm|y)P(y) for test pic,
        # where f1,f2,...,fm take values in test pic
        # that y is our prediction
        # P(y)=c(label=y)/total training data
        # P(f1,f2,...,fm|y)=P(f1|y)P(f2|y)...P(fm|y)
        # P(f1|y)=c(f1=x,label=y)/c(f1=0,1,label=y) x given in test pic, right side given by training

        bestKAccuracy = -1  # highest accuracy so far on validation set with smoothing factor k
        cPrior = util.Counter()  # the probability of this label showing up in the training set
        cConditionalProb = util.Counter()  # c(fi=1, y)
        cCounts = util.Counter()  # c(fi=0,1, y)
        for i in range(len(trainingData)):  # go thru all training picture
            datum = trainingData[i]  # a list of values for training pic i features, each item [(col, row), val]
            label = trainingLabels[i]  # y- label for pic i
            cPrior[label] += 1  # c(y) - times label y show up in training set
            for feat, value in datum.items():  # go thru all [feat, val] for pic i
                cCounts[(feat, label)] += 1  # [((col, row), y), count]-> c(fi=1,0|y)
                if value > 0:
                    cConditionalProb[(feat, label)] += 1  # [((col, row),y), count]->c(fi=1|y)
        for k in kgrid:
            prior = util.Counter()
            conditionalProb = util.Counter()
            counts = util.Counter()
            for key, val in cPrior.items():  # [y1, c(y1)],...
                prior[key] += val  # copy whole list
            for key, val in cCounts.items():  # [(f1, y1), c(f1=0,1, y1)]...
                counts[key] += val  # copy whole list
            for key, val in cConditionalProb.items():  # [(f1, y1), c(f1=1,y1)]...
                conditionalProb[key] += val  # copy whole list
            # smoothing:
            for label in self.legalLabels:
                for feat in self.features:
                    conditionalProb[(feat, label)] += k  # [(f1,y1), c(f1=1,y1)+k]...
                    counts[(feat, label)] += 2 * k  # both feat= 0, 1 are smoothed; [(f1,y1), c(f1=0,1,y1)+2k]...
            # normalizing:
            prior.normalize()  # [y1, c(y1)/n], [y2, c(y2)/n],...
            for x, count in conditionalProb.items():  # [(f1, y1), c(f1=1,y1)+k]...
                conditionalProb[x] = count * 1.0 / counts[x]  # [(f1, y1), (c(f1=1,y1)+k)/(c(f1=0,1, y1)+2k)]...
                # if f1=0 in test pic, P(f1=0|y)=1-P(f1=1|y)
            self.prior = prior
            self.conditionalProb = conditionalProb
            # evaluating performance on validation set
            predictions = self.classify(validationData)
            accuracyCount = [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
            print "Performance on validation set for k=%f: (%.1f%%)" % (
            k, 100.0 * accuracyCount / len(validationLabels))
            if accuracyCount > bestKAccuracy:
                bestParams = (prior, conditionalProb, k)
                bestKAccuracy = accuracyCount
        self.prior, self.conditionalProb, self.k = bestParams

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = []  # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)  # log P(y, f1,f2,...,fm)
            guesses.append(posterior.argMax())  # y that maximize log P
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        "*** YOUR CODE HERE ***"
        logJoint = util.Counter()
        for label in self.legalLabels:
            logJoint[label] = math.log(self.prior[label])  # log [y1, c(y1)/n], [y2, c(y2)/n]
            for feat, value in datum.items():  # [(col, row), val]
                if value > 0:
                    logJoint[label] += math.log(self.conditionalProb[feat, label])  # [(f1, y1), (c(f1=1,y1)+k)/(c(f1=0,1, y1)+2k)]...
                else:
                    logJoint[label] += math.log(1 - self.conditionalProb[feat, label])  # [(f1, y1), 1-(c(f1=1,y1)+k)/(c(f1=0,1, y1)+2k)]...
        return logJoint  # log[c(y1)/n]+log[(c(f1=1,y1)+k)/(c(f1=0,1, y1)+2k)]+...

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"

        return featuresOdds




import numpy as np
# You need to build your own model here instead of using well-built python packages such as sklearn
from utils import UnigramFeature
import main
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZeor(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Add your code here!
        #raise Exception("Must be implemented")
        self.hated_rows = []
        self.non_hated_rows = []
        self.joint_prob_hated = None
        self.joint_prob_non_hated = None
        self.prob_hated = None
        self.prob_non_hated = None
        self.highest_hate_prob_list = {}
        self.highest_non_hate_prob_list = {}
        

    def fit(self, X, Y):
        # Add your code here!
        for i in range(len(X)):
            if Y[i] == 0:
                self.non_hated_rows.append(X[i])
                
            if Y[i] == 1:
                self.hated_rows.append(X[i])
                

        # for tokens, label in zip(X,Y):
        #     if label == 0:
        #         self.non_hated_rows[]
        #     if label == 1:
        #         self.hated_rows.append(tokens)
        #print(total_hated,total_non_hated,int(total_hated+total_non_hated))
        self.hated_rows = np.asarray(self.hated_rows)
        self.non_hated_rows = np.asarray(self.non_hated_rows)
        sum_hated = np.sum(self.hated_rows,axis=0)
        sum_non_hated = np.sum(self.non_hated_rows,axis=0)
        self.prob_hated = (len(self.hated_rows))/(len(self.hated_rows)+len(self.non_hated_rows))
        self.prob_non_hated = (len(self.non_hated_rows))/(len(self.hated_rows)+len(self.non_hated_rows))
        self.joint_prob_hated = np.zeros(shape=sum_hated.shape)
        self.joint_prob_non_hated = np.zeros(shape=sum_non_hated.shape)
        #for i in range(len(X)):
        
        self.joint_prob_hated = (sum_hated+1)/(np.sum(sum_hated)+len(sum_hated))
        self.joint_prob_non_hated = (sum_non_hated+1)/(np.sum(sum_non_hated)+len(sum_non_hated))
        
        

    def predict(self, X):
        # Add your code here!
        predict = np.zeros(shape= X.shape[0])
        #print()
        for i in range(len(X)):
            hated_prob = 0
            non_hated_prob = 0
            for j in range(len(X[i])):
                hated_prob += X[i][j]*np.log(self.joint_prob_hated[j])
                non_hated_prob += X[i][j]*np.log(self.joint_prob_non_hated[j])

            hated_prob_sentence = np.log(self.prob_hated)+hated_prob
            non_hated_prob_sentence = np.log(self.prob_non_hated)+non_hated_prob

            if hated_prob_sentence>non_hated_prob_sentence:
                predict[i] = 1
            else:
                predict[i] = 0
        return predict


# TODO: Implement this
class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        # Add your code here!
        #raise Exception("Must be implemented")
        self.weights = None

        

    def fit(self, X, Y):
        # Add your code here!
        self.weights = self.gradientDescent(X,Y,2000,0.9,0)
        #raise Exception("Must be implemented")
        
    
    def predict(self, X):
        # Add your code here!
        predict = np.zeros(shape= X.shape[0])
        for row in range(len(X)):
            pred_prob = self.sigmoid(np.dot(X[row].T, self.weights))
            predict[row] = np.round(pred_prob)
        return predict
        #raise Exception("Must be implemented")


    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def gradientDescent(self,X,Y,epoch,learning_rate,regularization):
        weights= np.zeros(shape = X.shape[1])
        n = X.shape[0]
        d = X.shape[1]
        for e in range(epoch):
            for row in range(n):
                #loss = self.sigmoid(np.dot(X[row].T, weights))-Y[row]
                #weight[0] = weight[0]+learning_rate*loss
                weights -= ((learning_rate)/(len(Y)))*(np.dot(X[row].T,(self.sigmoid(np.dot(X[row].T, weights))-Y[row])))+ (2*regularization/(len(Y)))*weights

        return weights












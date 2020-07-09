from nltk.tokenize import regexp_tokenize
import numpy as np
from nltk.corpus import stopwords
import nltk

# Here is a default pattern for tokenization, you can substitue it with yours
default_pattern =  r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """

def tokenize(text, pattern = default_pattern):
    """Tokenize senten with specific pattern
    
    Arguments:
        text {str} -- sentence to be tokenized, such as "I love NLP"
    
    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
    
    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    text = text.lower()
    return regexp_tokenize(text, pattern)


class FeatureExtractor(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass



class UnigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        self.unigram = {}
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue

                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        feature = np.zeros(len(self.unigram))
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                feature[self.unigram[text[i].lower()]] += 1
        #print(feature)
        return feature
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        
        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)
            

class BigramFeature(FeatureExtractor):
    """Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self):
        # Add your code here!
        #raise Exception("Must be implemented")
        self.bigram = {}
    def fit(self, text_set):
        # Add your code here!
        # raise Exception("Must be implemented")
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue

    def transform(self, text):
        # Add your code here!
        #raise Exception("Must be implemented")
        feature = np.zeros(len(self.unigram))
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                feature[self.unigram[text[i].lower()]] += 1
        #print(feature)
        return feature

    def transform_list(self, text_set):
        # Add your code here!
        #raise Exception("Must be implemented")
        pass

class CustomFeature(FeatureExtractor):
    """customized feature extractor, such as TF-IDF
    """
    def __init__(self):
        
        self.unigram = {}
        nltk.download('stopwords')
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        punctutation = [',','-',':','?','!','"','.',',','`','~']
        numbers =  [ str(i) for i in range(0,10000)]
        stop_words = stopwords.words('english')
        index = 0
        new_text_set = []
        for i in range(0,len(text_set)):
            temp_list = []
            for j in range(0, len(text_set[i])):
                if text_set[i][j] not in stop_words and text_set[i][j] not in punctutation and text_set[i][j] not in numbers:
                    temp_list.append(text_set[i][j])
                else:
                    continue
            if temp_list:
                bigram_list = zip(temp_list,temp_list[1:])
                t = []
                for k,v in bigram_list:
                    t.append((k,v))
                new_text_set.append(t)

        for i in range(0, len(new_text_set)):
            #print(new_text_set[i])
            for j in range(0, len(new_text_set[i])):
                #print(new_text_set[i][j])
                a,b = new_text_set[i][j]
                if (a.lower(),b.lower()) not in self.unigram:
                    self.unigram[(a.lower(),b.lower())] = index
                    index += 1
                else:
                    continue

                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        feature = np.zeros(len(self.unigram))
        for a,b in zip(text,text[1:]):
            if (a.lower(),b.lower()) in self.unigram:
                feature[self.unigram[(a.lower(),b.lower())]] += 1
        #print(feature)
        return feature
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        
        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)


        

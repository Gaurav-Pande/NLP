import math, random
import collections

################################################################################
# Part 0: Utility Functions
################################################################################

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    # preprocess the string
    padding = start_pad(n)
    result = []
    padded_string  = padding+text
    #print(padded_string)
    for i in range(len(text)):
        result.append((padded_string[i:i+n],text[i]))
    return result

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model


################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        # n is the length n context or n-gram
        # k is the smoothening parameter
        self.n = n
        self.k = k
        self.count = collections.defaultdict(int)
        self.ngram = collections.defaultdict(int)
        self.vocab_set = set()
        

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab_set

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        #print(text)
        res = ngrams(self.n,text)
        # update occurance of a context 
        for context,t in res:
            self.count[context] += 1
        # update ngram dictionary
        for pair in res: 
            self.ngram[pair] +=1
        # update vocab
        for char in text:
            self.vocab_set.add(char)



    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        if context not in self.count and self.k == 0:
            return 1.0/len(self.get_vocab())
        #print(self.ngram[(context,char)])
        #print(self.count[context])
        return (self.ngram[(context,char)] + self.k)/( self.count[context] + self.k*len(self.get_vocab()) )
       
        


    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        #random.seed(1)
        rand_num = random.random()
        sorted_vocab = sorted(self.get_vocab())
        prob = 0
        # for i=0
        if rand_num< self.prob(context,sorted_vocab[0]):
            return sorted_vocab[0]
        for i in range(1,len(sorted_vocab)):
            prob += self.prob(context,sorted_vocab[i-1])
            prob_curr_char = prob + self.prob(context,sorted_vocab[i])
            if    prob <= rand_num < prob_curr_char:
                return sorted_vocab[i]
            else:
                continue

        return ''



    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        #random.seed(1)
        starting_context = '~'*self.n
        generated_text = ""
        current_context = starting_context
        for i in range(length):
            random_char = self.random_char(starting_context)
            current_context +=random_char
            generated_text += random_char
            starting_context = current_context[i+1:i+1+self.n]

        return generated_text

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        result = 0
        res = ngrams(self.n,text)
        for context,t in res:
            #print(context,t)
            c_prob = self.prob(context,t)
            if c_prob == 0:
                result = float('inf')
                return result
            result +=  math.log(self.prob(context,t))
        #print(result,math.exp(result))
        result  = math.exp((-1/len(text))*result)
        #result  = 1.0/math.pow(2,result)
        return result

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.count = collections.defaultdict(int)
        self.ngram = collections.defaultdict(int)
        self.vocab_set = set()
        #self.lmbda = [1/(n+1)]*(n+1) 
        self.lmbda = [0.9,0.05,0.02,0.02,0.01]

    def get_vocab(self):
        return self.vocab_set

    def update(self, text):
        res = []
        # update occurance of a context
        for i in range(self.n+1):
            res.extend(ngrams(i,text)) 
        for context,t in res:
            self.count[context] += 1
        # update ngram dictionary
        for pair in res: 
            self.ngram[pair] +=1
        # update vocab
        for char in text:
            self.vocab_set.add(char)
        

    def prob(self, context, char):
        if context not in self.count and self.k == 0:
            return 1.0/len(self.get_vocab())
        prob = 0

        for i in range(self.n+1):
            temp = self.lmbda[i]*((self.ngram[(context[i:],char)] + self.k)/(self.count[context[i:]]+self.k*len(self.get_vocab())))
            prob += temp
        return prob
################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass
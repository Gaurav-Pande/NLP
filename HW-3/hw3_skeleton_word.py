import math, random
from typing import List, Tuple
import collections

################################################################################
# Part 0: Utility Functions
################################################################################

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return ['~'] * n

Pair = Tuple[str, str]
Ngrams = List[Pair]
def ngrams(n, text:str) -> Ngrams:
    text=text.strip().split()
    #print(text)
    ''' Returns the ngrams of the text as tuples where the first element is
        the n-word sequence (i.e. "I love machine") context and the second is the word '''
    padding = start_pad(n)
    padded_text = padding+text
    result = []
    #print(text)
    for i in range(len(text)):
        # flag = False
        # for j in range(i,i+n):
        #     if len(padded_text[j]) != 1:
        #         break
        #     else:
        #         flag = True
        # if flag:
        #     result.append((''.join(tuple(padded_text[i:i+n])),text[i]))
        # else:
        result.append((' '.join(tuple(padded_text[i:i+n])),text[i]))
    #print(result)
    return result




def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8') as f:
        model.update(f.read())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocab = set()
        self.count = collections.defaultdict(int)
        self.ngram = collections.defaultdict(int)

    def get_vocab(self):
        ''' Returns the set of words in the vocab '''
        return self.vocab

    def update(self, text:str):
        ''' Updates the model n-grams based on text '''
        res = ngrams(self.n,text)

        #print(res)
        # update occurance of a context 
        #print(res)
        for context,t in res:
            self.count[context] += 1
        # update ngram dictionary
        for pair in res: 
            self.ngram[pair] +=1
        # update vocab
        text  = text.strip().split()
        for word in text:
            self.vocab.add(word)


    def prob(self, context:str, word:str):
        ''' Returns the probability of word appearing after context '''
        if context not in self.count and self.k == 0:
            return 1.0/len(self.get_vocab())
        #print(self.ngram[(context,char)])
        #print(self.count[context])
        # print((context,word))
        # print(context)
        # print(self.ngram)
        # print(self.count)
        #print("in prob calculation")
        #print((context,word, self.k))
        if (context,word) not in self.ngram:
            value1 = 0.0
        else:
            value1 = self.ngram[(context,word)]

        if context not in self.count:
            value2 = 0.0
        else:
            value2 = self.count[context]
        return (value1 + self.k) / (value2 + self.k*len(self.get_vocab()) )
        

    def random_word(self, context):
        ''' Returns a random word based on the given context and the
            n-grams learned by this model '''
#         random.seed(1)
        rand_num = random.random()
        sorted_vocab = sorted(self.get_vocab())
        prob = 0
        # for i=0
        #print(context, sorted_vocab)

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
        ''' Returns text of the specified word length based on the
            n-grams learned by this model '''
        starting_context = ['~'] * self.n
        starting_context = " ".join(starting_context)
        generated_text = ""
        current_context = starting_context
        #print(length, self.n)
        for i in range(length):
            random_char = self.random_word(starting_context)
            #print("generating word no", i, random_char)
            current_context += " " + random_char
            generated_text = generated_text + " " + random_char
            #print(current_context) 
            new_list = current_context
            temp_list  = new_list.split(" ")
            #current_context = current_context.strip().split()
            starting_context = temp_list[i+1:i+1+self.n]
            #print(starting_context)
            if '' in starting_context:
                starting_context = "".join(starting_context)
            else:
                starting_context = " ".join(starting_context)
            #starting_context = current_context[i+1:i+1+self.n]
            

        return generated_text[1:]

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        result = 0
        res = ngrams(self.n,text)
        #print(res)
        text=text.strip().split()
        #print(text)
        count_ngrams_dict = collections.Counter(res)
        for context,t in res:
            #print(context,t)
            c_prob = self.prob(context,t)
            if c_prob == 0.0:
                result = float('inf')
                return result
            result +=  math.log(self.prob(context,t))
        #print(result,math.exp(result))
        result  = math.exp(-result/len(text))
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
        self.lmbda = [0.1,0.3,0.6]

    def get_vocab(self):

        return self.vocab_set

    def update(self, text:str):
        
        res = []
        # update occurance of a context
        for i in range(self.n+1):
            #print(ngrams(i,text))
            res.extend(ngrams(i,text)) 
        
        for context,t in res:
            self.count[context] += 1
        # update ngram dictionary
        for pair in res: 
            self.ngram[pair] +=1
        # update vocab
        text = text.strip().split()
        for char in text:
            self.vocab_set.add(char)

    def prob(self, context:str, word:str):
        if context not in self.count and self.k == 0:
            return 1.0/len(self.get_vocab())
        prob = 0
        context =  context.split(" ")
        for i in range(self.n+1):
            #print('contextsad',context)
            if len(context) == 1:
                new_context = ''.join(context)
            else:
                new_context =  " ".join(context[self.n-i:self.n])
            #print(new_context,'da',len(new_context))
            if (new_context,word) not in self.ngram:
                value1 = 0.0
            else:
                value1 = self.ngram[(new_context,word)]

            if new_context not in self.count:
                value2 = 0.0
            else:
                value2 = self.count[new_context]
            #print('context',new_context)
            #print("vocab",self.get_vocab())
            temp = self.lmbda[i]*((value1 + self.k)/(value2+self.k*len(self.get_vocab())))
            prob += temp
        return prob

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass
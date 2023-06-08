

# Program to measure the similarity between 
# two sentences using cosine similarity.
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, SyllableTokenizer
from dataclasses import dataclass

from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer
  
# X = input("Enter first string: ").lower()
# Y = input("Enter second string: ").lower()
@dataclass
class CosineScorerConfig(FairseqDataclass):
    pass


@register_scorer("cosine", dataclass=CosineScorerConfig)
class CosineScorer(BaseScorer):
    def __init__(self, args):
        super(CosineScorer, self).__init__(args)
        try:
            import nltk
        except ImportError:
            raise ImportError("Please install nltk to use Cosine scorer")

        self.nltk = nltk
  

    def add_string(self, ref, pred):
        self.ref=ref
        self.pred=pred

    def score(self, tokenizer):  
        if tokenizer=='word':
            # tokenization
            X_list = word_tokenize(self.ref) 
            Y_list = word_tokenize(self.pred)
        else:
            syllable=SyllableTokenizer()
            X_list = syllable.tokenize(self.ref) 
            Y_list = syllable.tokenize(self.pred)

        
        # sw contains the list of stopwords
        sw = stopwords.words('english') 
        l1 =[];l2 =[]
        
        # remove stop words from the string
        X_set = {w for w in X_list if not w in sw} 
        Y_set = {w for w in Y_list if not w in sw}

        # form a set containing keywords of both strings 
        rvector = X_set.union(Y_set) 
        for w in rvector:
            if w in X_set: l1.append(1) # create a vector
            else: l1.append(0)
            if w in Y_set: l2.append(1)
            else: l2.append(0)
        c = 0
        
        # cosine formula 
        for i in range(len(rvector)):
                c+= l1[i]*l2[i]
        if float((sum(l1)*sum(l2))**0.5)!=0:
            cosine = c / float((sum(l1)*sum(l2))**0.5)
        else:
            cosine =0.0
        return cosine
    
    def result_string(self, tokenizer):
        return self.score(tokenizer).format()
   
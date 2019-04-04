#LDA code - Sandeep 3.28.19

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import RegexpTokenizer # remove stop words
tokenizer = RegexpTokenizer(r'\w+')
nltk.download('stopwords')    
from nltk.corpus import stopwords

class LdaCgsIteration():
    
    '''
    Implement LDA using Collapsed Gibbs Implementation Method
    
    input is a list wordDocMat, niter = 1200 , burn = 200, k = 5, 
                 alpha = 0, beta = 0

    '''
    stop_words = stopwords.words('english')
    
    def __init__(self, dat, niter=100, burn=20, n_topics = 5, 
                 alpha=50,beta=0.01, n_topwords = 10, stopwrds=stop_words):

        self.dat = dat
        self.niter = niter
        self.burn = burn
        self.n_topics = n_topics
        self.alpha = alpha/n_topics
        self.beta = beta
        self.stopwrds=stopwrds
        self.n_topwords = n_topwords

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def wordTokenize(self,txt):
        wrlist = tokenizer.tokenize(txt.lower())
        return wrlist
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def removeStopwords(self): 
        '''
        collects terms and removes stop words
        '''
        tkn_doc = []
        for i,k in enumerate(self.dat):
            #print(k)
            wrlist = self.wordTokenize(k)
            tkn_doc.append([k for k in wrlist if not k in self.stopwrds])
        return tkn_doc
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def removeDigits(self,terms):
        '''
        Remove digits from words
        '''
        p = re.compile(r'\d+')
        return [terms[i] for i in range(len(terms)) 
                if p.findall(terms[i])==[]]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def termsDoc(self): 
        '''
        remove stop word from the list
        and create a dictionary of doc-word counts
        '''
        doct = dict()
        
        if self.stopwrds is not None:
            tkn_doc = self.removeStopwords()
        else:
            tkn_doc = self.dat
         
        for i,k in enumerate(tkn_doc):  
            cnt = dict() 
            for w in k:
                if w in cnt.keys():
                    cnt[w] += 1
                else:
                    cnt[w] = 1
            doct[i] = cnt
        return doct
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setofterms(self):
        doct=self.termsDoc()
        terms = []

        for i in range(len(doct)):
            for j in doct[i].keys():
                terms.append(j)

        terms = list(set(terms))
        termClean = self.removeDigits(terms)
        return termClean

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def termDocMatrix(self):    
        '''
        #Creating a term-document matrix
        #A. Create a list of terms across the whole corpus
        #B. Remove alphanumeric terms
        '''
        
        doct = self.termsDoc()
        terms = []
        
        for i in range(len(doct)):
            for j in doct[i].keys():
                terms.append(j)
            
        terms = list(set(terms))
        
        termClean = self.removeDigits(terms)
        
        # Obtain document count and term count
        len_t = len(termClean)
        len_d = len(doct)

        term_doc = np.empty((len_t,len_d), dtype=np.int)

        # Term-Document Frequency Matrix 
        # Columns-->Document; Rows --> Terms
        # E.g. term_doc[:,0] -- Extracting 1st document

        for i in range(len_d):
            for t,word in enumerate(termClean):
                if word in doct[i].keys():
                    term_doc[t,i]=doct[i].get(word)
                else:
                    term_doc[t,i] = 0
        
        return term_doc
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialization of CGS

    def cgsIntialization(self):
        
        wordDocMat = self.termDocMatrix(self.dat)
        k = self.n_topics
        
        '''
        Input: 
            k = # of topic (int)
            chain = # of different starting point for the iterations (int)
            wordDocMat = Word-document matrix -- [V X d] array

        Output:
            n_t_k - number of times term 't' in topic 'k' - [V X k] array
            n_k_m - number of times topic 'k' in document 'd' - [d X k] array
            n_k - number of words in each topic
            n_m - length of document
            k - number of topics 

        '''

        len_t = wordDocMat.shape[0]  # number of words
        len_d = wordDocMat.shape[1]  # number of docs
        mat_V_k = np.empty((len_d,len_t,k), dtype = np.int)  #shape==>(doc, words, topic)

        for doc in range(len_d):
            for index, val in enumerate(wordDocMat[:,doc]):
                if  val!= 0:
                    mat_V_k[doc,index,:] = np.random.multinomial(1,[1./k]*k) 
                else:
                    mat_V_k[doc,index,:] = 0    
        
        return mat_V_k
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~
    def cgsIteration(self): #, wordDocMat, niter , burn, n_topics, 
                 #alpha, beta):
    
        '''
        Defintion: Iteration of the CGS 

        Input: 

            Output returned from cgsIntialization
            niter - total number of iterations
            burn -  number of iterations to discard
            alpha - hyperparameter influencing the shape of the Dirichlet priors
            beta -  " 
            wordDocMat - term-document matrix (V x d)

        Output: 

            finalThetaMK - array [d X k] - distribution of topic in a document
                                           (averaged over multiple iterations)
            finalphiTK -  array [t X k ] - distribution over word for each topic
                                           (averaged over multiple iterations) 
            mat_V_k - [d X t X k] - topic assignment for each for all document

        '''

        intIter = 0
        avgSmp = self.niter-self.burn
        k = self.n_topics
        wordDocMat = self.termDocMatrix()
        
        mat_V_k = self.cgsIntialization()

        n_k_m = np.sum(mat_V_k, axis=1)  #{doc, topics}
        n_t_k = np.sum(mat_V_k, axis=0)  #{words,topics}
        n_k = np.sum(n_t_k, axis=0)      #{words}
        n_m = np.sum(n_k_m, axis=1)      #{topics}

        lenM = mat_V_k.shape[0]
        lenT = mat_V_k.shape[1]

        thetaMK = np.empty((lenM,k,avgSmp), dtype = np.float) 
        phiTK = np.empty((lenT,k,avgSmp), dtype = np.float)

        while intIter < self.niter:

            for doc in range(lenM):   # each document
                for index, val in enumerate(wordDocMat[:,doc]): # each word
                    if  val != 0:   

                        # check if word is in ...
                        # subtracting its count from the ...
                        # topic assigned to the word in initialization;

                        n_k_m[doc,:] = n_k_m[doc,:] - mat_V_k[doc,index,:]
                        n_t_k[index,:] = n_t_k[index,:]-mat_V_k[doc,index,:]
                        n_k = n_k - mat_V_k[doc,index,:]
                        n_m[doc] = n_m[doc] - 1 

                        prob = np.empty(k, dtype=np.float)

                        for topic in range(k):
                            rhst1 = (n_k_m[doc,topic] + self.alpha)/((n_m[doc] + k*self.alpha)-1) 
                            rhst2 = (n_t_k[index,topic] + self.beta)/(n_k[topic] + lenT*self.beta)

                            prob[topic] = rhst1*rhst2

                        prob_new = prob/(np.sum(prob))

                        mat_V_k[doc,index,:] = np.random.multinomial(1,prob_new) 

                        n_k_m[doc,:] = n_k_m[doc,:] + mat_V_k[doc,index,:]                                                    
                        n_t_k[index,:] = n_t_k[index,:] + mat_V_k[doc,index,:]
                        n_k = n_k + mat_V_k[doc,index,:]
                        n_m[doc] = n_m[doc] + 1

            #print(intIter)

            # calculate the theta & phi parameters     
            
            if intIter >= self.burn:
                n_k_m1 = np.sum(mat_V_k, axis=1)  #{doc, topics}
                #print(intIter)
                n_t_k1 = np.sum(mat_V_k, axis=0)  #{words,topics}
                n_k1 = np.sum(n_t_k, axis=0)      # words count by topic
                n_m1 = np.sum(n_k_m, axis=1)      # length of document

                dim3 = intIter - self.burn

                for it1 in range(lenM):
                    thetaMK[it1,:, dim3] = (n_k_m1[it1,:] + self.alpha)/(n_k_m1[it1,:].sum(axis=0) + k*self.alpha)

                for it2 in range(k):
                    phiTK[:,it2, dim3] =  (n_t_k1[:,it2] + self.beta)/(n_t_k1[:,it2].sum(axis=0) + lenT*self.beta)

            intIter += 1

        finalThetaMK = np.mean(thetaMK, axis=2)
        finalphiTK = np.mean(phiTK, axis=2)

        return (finalThetaMK,finalphiTK,mat_V_k)
    

    #~~~~~~~~~~~~~~~~~~~~~~
    
    def execCGS(self, chain=1):
        '''
        Input:

        chain - number of different starting point 
                covering the starting distributions. 
                Recommended practise in CGS sampling.
        Output:

        Output from cgsIteration

        finalThetaMK - array [d X k] - distribution of topic in a document
                                      (averaged over multiple iterations within one chain)
        finalphiTK -  array [t X k ] - distribution over word for each topic
                                      (averaged over multiple iterations) 
        mat_V_k - [d X t X k] - topic assignment for each for all document

        '''
        chainIter = {} #to store output from each chain separately

        for i in range(chain):
            chainIter[i] = self.cgsIteration()

        return(chainIter)

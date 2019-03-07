# coding: utf-8
# ## Topic Model - LDA using Collapsed Gibbs Sampling
# Sandeep Shetty  
# 12/10/2018
 
import pandas as pd
import numpy as np

# Initialization of CGS

def cgsIntialization(k, wordDocMat):
    
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
    return(mat_V_k)


def cgsIteration(wordDocMat, niter = 1200 , burn = 200, k = 5, 
                 alpha = 0, beta = 0):
    
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
    avgSmp = niter-burn

    mat_V_k = cgsIntialization(k, wordDocMat)
     
    n_k_m = np.sum(mat_V_k, axis=1)  #{doc, topics}
    n_t_k = np.sum(mat_V_k, axis=0)  #{words,topics}
    n_k = np.sum(n_t_k, axis=0)      #{words}
    n_m = np.sum(n_k_m, axis=1)      #{topics}

    len_t = wordDocMat.shape[0]  # number of words
    len_d = wordDocMat.shape[1]  # number of docs

    lenM = mat_V_k.shape[0]
    lenT = mat_V_k.shape[1]
     
    thetaMK = np.empty((lenM,k,avgSmp), dtype = np.float) 
    phiTK = np.empty((lenT,k,avgSmp), dtype = np.float)
     
    while intIter < niter:
        
        for doc in range(len_d):   # each document
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
                        rhst1 = (n_k_m[doc,topic] + alpha)/((n_m[doc] + k*alpha)-1) 
                        rhst2 = (n_t_k[index,topic] + beta)/(n_k[topic] + lenT*beta)

                        prob[topic] = rhst1*rhst2

                    prob_new = prob/(np.sum(prob))

                    mat_V_k[doc,index,:] = np.random.multinomial(1,prob_new) 

                    n_k_m[doc,:] = n_k_m[doc,:] + mat_V_k[doc,index,:]                                                    
                    n_t_k[index,:] = n_t_k[index,:] + mat_V_k[doc,index,:]
                    n_k = n_k + mat_V_k[doc,index,:]
                    n_m[doc] = n_m[doc] + 1
        
        #print(intIter)
        
        # calculate the theta & phi parameters     
        if intIter >= burn:
            n_k_m1 = np.sum(mat_V_k, axis=1)  #{doc, topics}
            print(intIter)
            n_t_k1 = np.sum(mat_V_k, axis=0)  #{words,topics}
            n_k1 = np.sum(n_t_k1, axis=0)      # words count by topic
            n_m1 = np.sum(n_k_m1, axis=1)      # length of document
            
            dim3 = intIter - burn
            
            for it1 in range(lenM):
                thetaMK[it1,:, dim3] = (n_k_m1[it1,:] + alpha)/(n_k_m1[it1,:].sum(axis=0) + k*alpha)
        
            for it2 in range(k):
                phiTK[:,it2, dim3] =  (n_t_k1[:,it2] + beta)/(n_t_k1[:,it2].sum(axis=0) + lenT*beta)
                
        intIter += 1
        
    finalThetaMK = np.mean(thetaMK, axis=2)
    finalphiTK = np.mean(phiTK, axis=2)
            
    return((finalThetaMK,finalphiTK,mat_V_k))          

# ### Setting up the LDA-CGS execution


def execCGS(chain):
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
        chainIter[i] = cgsIteration(niter = 4, 
                                    burn = 2, 
                                    k = 5, 
                                    alpha = 50/5, #(=50/k)
                                    beta = 0.01, 
                                    wordDocMat=term_doc)
    
    return(chainIter)


# ### Running the CGS

chain_out = execCGS(chain=1)

# ### Simple reporting


for i in range(5):
    strin1 = 'Topic'+str(i)
    data= pd.DataFrame(np.array(termClean),chain_out[0][1][:,i], columns=[strin1])
    print(data.drop_duplicates().sort_index(ascending=False).iloc[10:20])
    print("------------------")


##### END OF LDA-CGS

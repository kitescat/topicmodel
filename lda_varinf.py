# Code for LDA-Variational Inference
# Sandeep Shetty - 1.18.19
# Pseudocode followed from Blei et al (2003)
# Status - figuring it out (v1)

'''

*** Input 
 Term list
 Number of topics - k 
 Raw-word-matrix (not unique words) - wordDocMat
 Tolerance values - tol

*** Output
 LDA paramters - Beta (probability..)

'''
# Global-ish variables

doc_psi = np.empty(shape=[len_d,len_t,k], dtype=np.float)
doc_gamma = np.empty(shape = [len_d, k], dtype=np.float)

# Calculation for each document individually
for docum in range(len_d):

    # 2-D matrix to holding 'psi' (document X word X topic)
    pi_tk = np.empty(shape=[len_t,k],dtype=np.float)
    
    # 1-D array for 'gamma' (document X topic)
    gamma_dk = np.empty(k,dtype=np.float)

    # 2-D to hold the word probabilities (word X topic)
    psi_n = np.empty(shape=[len_t,k], dtype=np.float)
    
    # Pre-calculate some terms
    # actual words in the document 
    docWord = [w for w in wordDocMat[:,docum] if w != 0] 
    
    # unique words
    setdocWord = list(set(docWord)) 
    
    # length of the document selected
    len_docu = len(docWord) 
    
    #INITIALIZATION
    
    # Intialize parameter "phi" in word-topic model for each document
    phi = 1./k

    # Initialize assignment (z_{d,n})
    for index, val in enumerate(wordDocMat[:,docum]):
        if  val!= 0:
            pi_tk[index,:] = np.random.multinomial(1,[phi]*k) 
        else:
            pi_tk[index,:] = 0 

    # Intialize "gamma" in per topic  for each document
    gamma_k = [alpha + len_docu/k]*k

    
    #ITERATION
    
    diff = True
    
    while diff:
        for index,word in enumerate(wordDocMat[:,docum]):
            if word!=0:
                
                # Frequency of the selected word 
                countWordinDoc = wordDocMat[index,docum] 
                
                # Actual word at that location
                wordAtN = termClean[index]
                
                # Count of all topic assignment in doc
                sumTopic = np.sum(pi_tk, axis = 0)    
                prb_ij = np.empty(k,dtype=np.float)
                 
                # Calculating beta_ij...towards Prob(wn=w|zn=i)  
                for k1 in range(k):
                    beta_ij = 0                          
                    for nindex in range(pi_tk.shape[0]):
                        if pi_tk[nindex,k1]==1 and termClean[nindex] == wordAtN:
                            beta_ij += 1
                            
                    # Prob(wn=w|zn=i)
                    prb_ij[k1] = (beta_ij/countWordinDoc+1)/ \
                                 ((sumTopic[k1])/len_docu)* \
                                 digamma(gamma_k[k1])
                            
                # Normalize probability    
                psi_n[index,:] = prb_ij/prb_ij.sum()  
                
            else:
                psi_n[index,:] = 0
                
        # Updating Gamma    
        newgamma = alpha + np.sum(psi_n,axis=0)
        
        # Updating assignment
        for index, val in enumerate(wordDocMat[:,docum]):
            if  val!= 0:
                pi_tk[index,:] = np.random.multinomial(1,psi_n[index,:]) 
            else:
                pi_tk[index,:] = 0 
        
        # Calculating "while" condition 
        diff = (abs(gamma_k - newgamma)>tolerance).all()
        
        # Reassigning
        gamma_k = newgamma 
    
    # Stacking word probabilities for {doc,word,topic}    
    doc_psi[docum,:,:] = psi_n
    
    # Stacking gamma for document
    doc_gamma[docum,:] = newgamma
    print('doc #', docum)


### M-step: Estimating $\beta_{k}$,  the distribution over the vocabulary (topic distribution)


final_b_k = np.empty([len_t,k],dtype=np.float)
b_ij = np.empty([len_t,len_d,k],dtype=np.float)

for doc in range(len_d):
    for index in range(len_t):
        if wordDocMat[index,doc] != 0:
            b_ij[index,doc,:] = psi_n[index,:]
        else:
            b_ij[index,doc,:] = 0

# almost final estimate of beta_{ij}
final_b_k = b_ij.sum(axis=1)

# Normalize (final) estimate of beta_{ij}
# Distribution of topics over words

final_b_k = final_b_k/(np.sum(final_b_k, axis=0))

#Output
#------
for i in range(3):
    strin1 = 'Topic'+str(i)
    data= pd.DataFrame(np.array(termClean),final_b_k[:,i], columns=[strin1])
    print(data.drop_duplicates().sort_index(ascending=False).iloc[0:20])
    print("------------------")

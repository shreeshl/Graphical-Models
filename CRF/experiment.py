'''
Use this file to answer question 5.1 and 5.2
'''
from __future__ import division
import numpy as np
import time
import matplotlib.pyplot as plt
from code.crf import CRF
from data_load import data_load 
import string
from collections import defaultdict

L = list(string.letters[26:])
Xtrain, Ytrain, Xtest, Ytest = data_load()

def five_one():
    '''implement your experiments for question 5.1 here'''
    data_range = range(100,900,100)
    lhood = []
    error = []
    for i in data_range:
        crf = CRF(L,321)
        crf.fit(Ytrain[:i], Xtrain[:i])
        # print "Data until %d trained. Time Taken : %.3f sec"%(i, time.time()-t)        
        lhood.append(crf.log_likelihood(Ytest, Xtest))
        avg_error = 0
        total_elt = 0
        for xi, yi in zip(Xtest, Ytest):
            ypred = crf.predict(xi)
            total_elt += len(ypred)
            avg_error += np.sum(ypred!=np.array(list(yi)))
        error.append(avg_error/total_elt)

    plt.plot(data_range, lhood)
    plt.xlabel('No. Training Instances')
    plt.ylabel('Average Test Set Log Likelihood')
    plt.savefig('lhood.png', bbox_inches='tight')
    plt.show()

    plt.plot(data_range, error)
    plt.xlabel('No. Training Instances')
    plt.ylabel('Average Test Error')
    plt.savefig('error.png', bbox_inches='tight')
    plt.show()


def five_two():
    '''implement your experiments for question 5.2 here'''
    words = defaultdict(list)

    min_seq_len = 100
    for x,y in zip(Xtest,Ytest):
        min_seq_len = min(min_seq_len, len(y))
        words[len(y)].append((x,y))

    seq_per_collection = 3
    upto_seq_len = 20

    seq_len = 1
    while seq_len<=upto_seq_len:
        count = len(words[seq_len])

        if count>=seq_per_collection:
            seq_len+=1
            continue

        if seq_len==1:
            for i in range(seq_per_collection-count):
                data = words[seq_len][i]
                x = np.vstack((data[0],data[0]))
                y += data[1]
                words[seq_len].append([x,y])

        else:
            for i in range(seq_per_collection-count):
                if seq_len%2==1 : data1, data2 = words[seq_len//2][i], words[seq_len - seq_len//2][i]
                else : data1 = data2 = words[seq_len/2][i]   
                x, y = np.vstack((data1[0],data2[0])), data1[1] + data2[1]
                words[seq_len].append([x,y])

        seq_len+=1

    
    crf = CRF(L,321)
    crf.fit(Ytrain, Xtrain)
    
    timekeeper = defaultdict(float)
    no_inference_repetition = 3
    for k in words:
        for r in range(no_inference_repetition):
            for i in range(seq_per_collection):
                t = time.time()
                crf.predict(words[k][i][0])
                timekeeper[k]+=time.time()-t
        
        timekeeper[k] = timekeeper[k]/(seq_per_collection*no_inference_repetition)
    
    plt.plot(timekeeper.keys(), timekeeper.values())
    plt.xlabel('Sequence Length')
    plt.ylabel('Time Taken')
    plt.savefig('inference.png', bbox_inches='tight')
    plt.show()
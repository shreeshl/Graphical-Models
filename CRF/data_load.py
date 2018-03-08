import numpy as np

def to_list(y):
    y = y.replace("\n", "")
    return [i for i in y]

def data_load():
    
    Ytrain = [to_list(i) for i in open("../data/train_words.txt")]
    Xtrain = [np.loadtxt("../data/train_img{}.txt".format(i),ndmin=2) for i in range(1,len(Ytrain) + 1)]

    Ytest = [to_list(i) for i in open("../data/test_words.txt")]
    Xtest = [np.loadtxt("../data/test_img{}.txt".format(i),ndmin=2) for i in range(1,len(Ytest) + 1)]
    
    return Xtrain, Ytrain, Xtest, Ytest
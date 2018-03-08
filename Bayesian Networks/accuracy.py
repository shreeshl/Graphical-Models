from bn import BayesNet
from bn_custom import BayesNetCustom
import numpy as np

def mean_variance():
    
    data_directory = '/Users/shreesh/Academics/CS688/HW1/HW01/data/data-'
    
    bn = BayesNetCustom()
    accuracy = []
    
    for i in range(1,6):
        test_data  = np.loadtxt(data_directory+'test-'+str(i)+'.txt', delimiter=",")
        train_data = np.loadtxt(data_directory+'train-'+str(i)+'.txt', delimiter=",")
        bn.fit(train_data)
        predictions = bn.predict_hd(test_data.copy())
        accuracy.append(np.mean(predictions == test_data[:,-1]))
    print np.mean(accuracy), np.std(accuracy)
    return
from __future__ import division
import numpy as np
from bn import BayesNet



class BayesNetCustom(BayesNet):
    """
    This class implements a Bayes Net
    """
    def __init__(self):
        '''
        This method will run upon initialization of the Bayes Net class
        You can structure this class in whatever way seems best.

        The class will need to support four methods by the end of the assignment.
            - fit: sets the parameters of the Bayes Net based on data
            - predict_hd: predicts a heart disease value, based on observed data
            - get: returns a given real-valued parameter
            - set: set the value of a parameter

        input:
            - None
        returns:
            - None
        '''
        self.target_condition = {'A':'', 'G':'', 'CH':'A,G', 'BP':'G', 'HD':'CH,BP,A', 'CP':'HD,A,G', 'EIA':'HD,A,G', 'ECG':'HD,G,A', 'HR':'HD,BP,A'}
        BayesNet.__init__(self, self.target_condition)
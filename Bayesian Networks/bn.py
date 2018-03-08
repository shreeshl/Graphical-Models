from __future__ import division
import numpy as np
from collections import defaultdict
import itertools

values = {
            'A': [1,2,3], 'G': [1,2], 'CH': [1,2], 'BP': [1,2],
            'HD': [1,2], 'CP': [1,2,3,4], 'EIA': [1,2], 'ECG': [1,2],'HR' : [1,2]
        }

class CPT(object):
    def __init__(self, target_variable, condition_variables):
        
        self.target_variable = target_variable
        if condition_variables!='':
            self.condition_variables = condition_variables.split(',')
            self.has_cvs = True
        else:
            self.condition_variables = []
            self.has_cvs = False
        
        self.table = defaultdict(dict)
        self.populate()

    def populate(self):
        
        prob = 1/len(values[self.target_variable])
        self.possible_keys = self.all_possible_keys()
        for key in self.possible_keys:
            key = tuple(sorted(key.items()))
            for tv_val in values[self.target_variable]:
                self.table[key][self.target_variable,tv_val] = prob
    
    def all_possible_keys(self):
        
        if self.has_cvs:
            dicts = {cv:values[cv] for cv in self.condition_variables}
            return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))     
        
        return [{'NO_CV':'NO_CV'}]

        

class BayesNet(CPT):
    """
    This class implements a Bayes Net
    """
    def __init__(self, target_condition = {'A':'', 'G':'', 'CH':'A,G', 'BP':'G', 'HD':'CH,BP', 'CP':'HD', 'EIA':'HD', 'ECG':'HD', 'HR':'HD,BP,A'}):
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
        self.data = {}
        self.values = {
            'A': [1,2,3], 'G': [1,2], 'CH': [1,2], 'BP': [1,2],
            'HD': [1,2], 'CP': [1,2,3,4], 'EIA': [1,2], 'ECG': [1,2],'HR' : [1,2]
        }
        self.var2idx = {
            'A': 0, 'G': 1, 'CP': 2, 'BP': 3,
            'CH': 4, 'ECG': 5, 'HR': 6, 'EIA': 7,'HD': 8
        }
        self.map_for_key = {
            'L' : 1, 'H' : 2, 'F': 1, 'M': 2, 'N': 1, 'A': 2, 'Y': 2, '<45' : 1, '45-55': 2, '>=55':3, 'Typical' : 1, 'Atypical' : 2, 'Non-Anginal' : 3, 'None':4, 'Non-Angial' : 3,
            'Normal': 1, 'Abnormal' : 2
        }
        
        self.target_condition = target_condition

        for target, condition in self.target_condition.items():
            self.data[target] = CPT(target, condition)

    def variables_cleanup(self, target_variable, condition_variables):
        
        if condition_variables=={}:
            condition_variables = {'NO_CV':'NO_CV'}
        else:
            for c, v in condition_variables.items():
                if v in self.map_for_key: condition_variables[c] = self.map_for_key[v]
        
        for c, v in target_variable.items():
            if v in self.map_for_key: target_variable[c] = self.map_for_key[v]
        
        return target_variable, condition_variables

        

    def get(self, target_variable, condition_variables):
        '''
        This method does a lookup of a parameter value in your BayesNet
        For instance, you might want to lookup of p_theta(HD=N | CH=L, BP=L)

        inputs:
            - target_variable and value:
                - a dictionary, such as {'HD':'N'}
            - condition_variables and values
                - a dictionary, such as {'CH':'L', 'BP':'L'}
        returns:
            - The parameter value, a real value within [0,1]
            - If there is a no such parameter in the model, return None
        '''

        target_variable, condition_variables = self.variables_cleanup(target_variable, condition_variables)
        
        tv = target_variable.keys()[0]
        key = tuple(sorted(condition_variables.items()))
        
        if key in self.data[tv].table:
            return self.data[tv].table[key][target_variable.items()[0]]
        return None

    def set(self, target_variable, condition_variables, value):
        '''
        This method sets a parameter value in your BayesNet to value

        After you call the method, the parameter should be set to value
        For instance, you might want to set p(HD|BP,CH) = .222

        inputs:
            - target_variable and value:
                - a dictionary, such as {'HD':'N'}
            - condition_variables and values
                - a dictionary, such as {'CH':'L', 'BP':'L'}
            - value:
                -  probability between 0 and 1
        returns:
            - None
        '''
        target_variable, condition_variables = self.variables_cleanup(target_variable, condition_variables)
        
        tv = target_variable.keys()[0]
        key = tuple(sorted(condition_variables.items()))
        if key in self.data[tv].table:
            self.data[tv].table[key][target_variable.items()[0]] = value
 
    def fit(self, data):
        '''
        This method sets the parameters of your BayesNet to their MLEs
        based on the provided data. The layout of the data array and the
        coding used is described in the hadout.

        input:
            - data, a numpy array with the schema described in the handout
        returns:
            - None
        '''
        for cpt in self.data.values():
            tv = cpt.target_variable
            cv = cpt.condition_variables
            count = defaultdict(lambda: defaultdict(int))
            for data_point in data:
                if len(cv)!=0 : key = {var:data_point[self.var2idx[var]] for var in cv}
                else: key = {'NO_CV':'NO_CV'}
                count[tuple(sorted(key.items()))][tv,data_point[self.var2idx[tv]]] += 1
            
            for key in count:
                total = sum(count[key].itervalues(), 0.0)
                cpt.table[key] = {k: count[key][k] / total for k, v in cpt.table[key].iteritems()}


    def calculate_joint_distribution(self, data_point, epsilon = 0):
        '''
        This method returns a joint probability distribution over all nodes
        of the network for a given configuration of variables denoted by data_point

        input
            - data_point, an array with the schema described in the handout
        output
            - joint probability distribution for given data_point
        '''
        total = 1.0
        for cpt in self.data.values():
            cv = cpt.condition_variables
            tv = cpt.target_variable
            if len(cv)!=0 : key = {var:data_point[self.var2idx[var]] for var in cv}
            else: key = {'NO_CV':'NO_CV'}
            key = tuple(sorted(key.items()))
            total *= cpt.table[key][tv,data_point[self.var2idx[tv]]]
         
        return total + epsilon


    def predict_hd(self, data):
        '''
        - input:
            - data. An array of shape (N,D). The layout of the data array and the
        coding used is described in the hadout.

        - returns:
            - the predictions for your data, a numpy array with shape = (N,)
        '''
        out = np.ones((data.shape[0],))
        probs = {}
        for i, data_point in enumerate(data):
            for hd_val in values['HD']:
                data_point[self.var2idx['HD']] = hd_val
                probs[hd_val] = self.calculate_joint_distribution(data_point)
            
            if probs[2]/(probs[2]+probs[1]) > .5: out[i] = 2
        
        return out
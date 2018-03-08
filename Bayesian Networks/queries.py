
def query_5_a(bn):
    '''
    Please write a method which returns the distribution over the query
    variable, CH.

    You should return a dictionary, {"H":p1, "L":p2} where p1 and p2 are
    the probabilities that the patient has high or low cholesterol.

    You should be able to use the get method from your BayesNet class to implement
    this function.

    inputs:
        - bn: a parameterized Bayes Net
    returns:
        - out: a dictionary of probabilities, {"H":p1, "L":p2}

    '''
    probs = {}
    data_point = [2,'M','None','L','CH','Normal','L','N','N']
    
    for i, data in enumerate(data_point):
        if data in bn.map_for_key: data_point[i] = bn.map_for_key[data]
    
    for ch_val in bn.values['CH']:
        data_point[bn.var2idx['CH']] = ch_val
        probs[ch_val] = bn.calculate_joint_distribution(data_point)
    
    total = probs[2] + probs[1]
    out =  {"H":probs[2]/total, "L":probs[1]/total}
    # cv  = {'A':2, 'G':'M'}
    # out =  {"H":bn.get({'CH':2},cv), "L":bn.get({'CH':1},cv)}

    return out


def query_5_b(bn):
    '''
    Please write a method which returns an answer to query 5b from the problem set
    input:
        - bn: a parameterized Bayes Net

    returns:
        answers, a dictionary with two keys, "H" and "L". "H" is the probability
        of high BP given the specified conditions. "L" is the probability
        of low BP, given the specified conditions
    '''
    
    probs = {}
    data_point = [2,'G','Typical','BP','H','Normal','H','Y','N']
    
    for i, point in enumerate(data_point):
        if point in bn.map_for_key: data_point[i] = bn.map_for_key[point]
    
    data_point_1 = data_point[:]
    data_point_2 = data_point[:]
    for bp_val in bn.values['BP']:
        data_point_1[bn.var2idx['BP']] = data_point_2[bn.var2idx['BP']] = bp_val
        data_point_1[bn.var2idx['G']] = 1
        data_point_2[bn.var2idx['G']] = 2
        probs[bp_val] = bn.calculate_joint_distribution(data_point_1) + bn.calculate_joint_distribution(data_point_2)
    
    total = probs[2] + probs[1]
    out =  {"H":probs[2]/total, "L":probs[1]/total}

    return out

from __future__ import division
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import fmin_bfgs, check_grad, fmin_l_bfgs_b

class CRF(object):

    def __init__(self,L,F):
        '''
        This class implements learning and prediction for a conditional random field.

        Args:
            L: a list of label types
            F: the number of features

        Returns:
            None
        '''

        #Your code must use the following member variables
        #for the model parameters. W_F should have dimension (|L|,F)
        #while W_T should have dimension (|L|,|L|). |L| refers to the
        #number of label types. The value W_T[i,j] refers to the
        #weight on the potential for transitioning from label L[i]
        # to label L[j]. W_F[i,j] refers to the feature potential
        #between label L[i] and feature dimension j.
        self.L = L
        self.F = F
        self.label_to_idx = {val:idx for idx, val in enumerate(L)}
        self.idx_to_label = {val:key for key, val in self.label_to_idx.iteritems()}
        np.random.seed(0)
        self.W_F = np.random.randn(len(L),F)
        self.W_T = np.random.randn(len(L),len(L))

        # self.W_F = np.zeros((len(L),F))
        # self.W_T = np.zeros((len(L),len(L)))
        self.gW_F = np.zeros(self.W_F.shape)
        self.gW_T = np.zeros(self.W_T.shape)

    def get_params(self):
        '''
        Args:
            None

        Returns:
            (W_F,W_T) : a tuple, where W_F and W_T are the current feature
            parameter and transition parameter member variables.
        '''
        return self.W_F, self.W_T

    def set_params(self, W_F, W_T):
        '''
        Set the member variables corresponding to the parameters W_F and W_T

        Args:
            W_F (numpy.ndarray): feature paramters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            None

        '''
        self.W_F = W_F
        self.W_T = W_T

    def energy(self, Y, X, W_F=None, W_T=None):
        '''
        Compute the energy of a label sequence

        Args:
            Y (list): a list of labels from L of length T.
            X (numpy.ndarray): the observed data. A an array of shape (T,F)
            W_F (numpy.ndarray, optional): feature parameters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            E (float): The energy of Y,X.
        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        
        energy = 0
        for i, yi in enumerate(Y):
            energy += np.dot(W_F[self.label_to_idx[yi]], X[i,:].T)
            if i<len(Y)-1: energy += W_T[self.label_to_idx[Y[i]], self.label_to_idx[Y[i+1]]]
        
        return -energy



    def log_Z(self, X, W_F=None, W_T=None):
        '''
        Compute the log partition function for a feature sequence X
        using the parameters W_F and W_T.
        This computation must use the log-space sum-product message
        passing algorithm and should be implemented as efficiently
        as possible.

        Args:
            X (numpy.ndarray): the observed data. A an array of shape (T,F)
            W_F (numpy.ndarray, optional): feature parameters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            log_Z (float): The log of the partition function given X

        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        
        omega = np.zeros(( X.shape[0], len(self.L)))
        omega[-1,:] = np.dot(W_F,X[-1,:].T)

        for i in range(X.shape[0]-2,-1,-1): 
            for j in range(len(self.L)):  
                omega[i,j] = logsumexp(W_T[j,:] + np.dot(W_F[j,:], X[i,:]) + omega[i+1,:])

        return logsumexp(omega[0, :])

    def predict_logprob2(self, X, W_F=None, W_T=None):
        '''
        Compute the log of the marginal probability over the label set at each position in a
        sequence of length T given the features in X and parameters W_F and W_T

        Args:
            X (numpy.ndarray): the observed data. A an array of shape (T,F)
            W_F (numpy.ndarray, optional): feature parameters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

       Returns:
           log_prob (numpy.ndarray): log marginal probabilties, a numpy array of type float w/ dimension (T, |L|)
           log_pairwise_marginals (numpy.ndarray): log marginal probabilties, a numpy array of type float w/ dimension (T - 1, |L|, |L|)
               - log_pairwise_marginals[t][l][l_prime] should represent the log probability of the symbol, l, and the next symbol, l_prime,
                 at time t.
               - Note: log_pairwise_marginals is a 3 dimensional array.
               - Note: the first dimension of log_pairwise_marginals is T-1 because
                       there are T-1 total pairwise marginals in a sequence in length T
        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        
        omega = np.zeros(( X.shape[0], len(self.L)))
        alpha = np.zeros(( X.shape[0], len(self.L)))
        
        for i in range(X.shape[0]-2,-1,-1): 
            omega[i, :] = logsumexp(W_T + np.dot(W_F, X[i+1,:].T) + omega[i+1, :], axis=1)

        for j in range(1, X.shape[0]):
            alpha[j, :] = logsumexp(W_T + np.dot(W_F, X[j-1,:].T) + alpha[j-1, :], axis=1)

        logz = logsumexp(np.dot(W_F, X[0,:].T) + omega[0, :])
        log_prob = omega + alpha + np.dot(X, W_F.T) - logz
        
        log_pairwise_marginals = np.zeros((X.shape[0]-1, len(self.L), len(self.L)))
        for i in range(X.shape[0]-1):
            for y in self.L:
                y = self.label_to_idx[y]
                log_pairwise_marginals[i, y, :] = omega[i+1,:] + alpha[i,y] + W_T[y,:] + np.dot(W_F[y,:], X[i,:].T) + np.dot(W_F, X[i+1,:].T) - logz

        return log_prob, log_pairwise_marginals

    def predict_logprob(self, X, W_F=None, W_T=None):
        '''
        Compute the log of the marginal probability over the label set at each position in a
        sequence of length T given the features in X and parameters W_F and W_T

        Args:
            X (numpy.ndarray): the observed data. A an array of shape (T,F)
            W_F (numpy.ndarray, optional): feature parameters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

       Returns:
           log_prob (numpy.ndarray): log marginal probabilties, a numpy array of type float w/ dimension (T, |L|)
           log_pairwise_marginals (numpy.ndarray): log marginal probabilties, a numpy array of type float w/ dimension (T - 1, |L|, |L|)
               - log_pairwise_marginals[t][l][l_prime] should represent the log probability of the symbol, l, and the next symbol, l_prime,
                 at time t.
               - Note: log_pairwise_marginals is a 3 dimensional array.
               - Note: the first dimension of log_pairwise_marginals is T-1 because
                       there are T-1 total pairwise marginals in a sequence in length T
        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        
        omega = np.zeros(( X.shape[0], len(self.L)))
        alpha = np.zeros(( X.shape[0], len(self.L)))
        alpha[0,:]  = np.dot(W_F,X[0,:].T)
        omega[-1,:] = np.dot(W_F,X[-1,:].T)
        
        for i in range(X.shape[0]-2,-1,-1): 
            for j in range(len(self.L)):  
                omega[i,j] = logsumexp(W_T[j,:] + np.dot(W_F[j,:], X[i,:]) + omega[i+1,:])

        for i in range(1, X.shape[0]):
            for j in range(len(self.L)):
                alpha[i,j] = logsumexp(W_T[:,j] + np.dot(W_F[j,:], X[i,:]) + alpha[i-1,:])
        
        logz = logsumexp(omega[0,:])

        log_prob = omega + alpha - np.dot(X,W_F.T) - logz
        log_pairwise_marginals = np.zeros((X.shape[0]-1, len(self.L), len(self.L)))
        
        for i in range(X.shape[0]):
            if i < X.shape[0]-1:
                for y in self.L:
                    y = self.label_to_idx[y]
                    for yprime in self.L:
                        yprime = self.label_to_idx[yprime]
                        log_pairwise_marginals[i, y, yprime] = omega[i+1,yprime] + alpha[i,y] + W_T[y,yprime] - logz

        return log_prob, log_pairwise_marginals



    def predict(self, X, W_F=None, W_T=None):
        '''
        Return a list of length T containing the sequence of labels with maximum
        marginal probability for each position in an input fearture sequence of length T.

        Args:
            X (numpy.ndarray): the observed data. A an array of shape (T,F)
            W_F (numpy.ndarray, optional): feature paramters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            Yhat (list): a list of length T containing the max marginal labels given X
        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        
        
        Yhat = []
        log_prob, _ = self.predict_logprob(X)
        log_prob = np.argmax(log_prob,axis=1)
        for item in log_prob:
            Yhat.append(self.idx_to_label[item])
        # your implementation here

        assert len(Yhat) == X.shape[0]

        return Yhat

    def log_likelihood(self, Y, X, W_F=None, W_T=None):
        '''
        Calculate the average log likelihood of N labeled sequences under
        parameters W_F and W_T. This must be computed as efficiently as possible.

        Args:
            Y (list): a list of length N where each element n is a list of T_n labels from L
            X (list): a list of length N where each element n is a feature array of shape (T_n,F)
            W_F (numpy.ndarray, optional): feature paramters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            mll (float): the mean log likelihood of Y and X
        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T

        
        mll = 0.0
        for x, y in zip(X,Y):
            # if len(x.shape)==1: x = x[None,:]
            mll -= self.energy(y,x,W_F,W_T) + self.log_Z(x,W_F,W_T)
        
        N = len(X)
        return mll/N
        # your implementation here

    def gradient_log_likelihood(self, Y, X, W_F=None, W_T=None):
        '''
        Compute the gradient of the average log likelihood
        given the parameters W_F, W_T. Your implementation
        must be as efficient as possible.

        Args:
            Y (list): a list of length N where each element n is a list of T_n labels from L
            X (list): a list of length N where each element n is a feature array of shape (T_n,F)
            W_F (numpy.ndarray, optional): feature paramters, a numpy array of type float w/ dimension (|L|,F)
            W_T (numpy.ndarray, optional): transition parameters, a numpy array of type float w/ dimension (|L|,|L|)

        Returns:
            (gW_F, gW_T) (tuple): a tuple of numpy arrays the same size as W_F and W_T containing the gradients

        '''
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        
        gW_F = self.gW_F
        gW_T = self.gW_T

        for xi,yi in zip(X,Y):
            # if len(xi.shape)==1: xi = xi[None,:]
            no_classes = W_T.shape[0]
            seq_len    = len(yi)

            log_prob, log_pairwise_marginals = self.predict_logprob(xi, W_F, W_T)

            # labels  = np.array([self.label_to_idx[yij] for yij in yi])
            # mask    = (np.tile(range(no_classes),(seq_len,1)).T==labels).astype('int')
            # mask1   = (np.tile(range(no_classes),(seq_len-1,1)).T==labels[:-1]).astype('int')
            # mask2   = (np.tile(range(no_classes),(seq_len-1,1)).T==labels[1:]).astype('int')            
            # gW_F += np.dot(mask,xi) - np.dot(np.exp(log_prob).T, xi)
            # gW_T += np.dot(mask1, mask2.T) - np.sum(np.exp(log_pairwise_marginals),axis=0)

            gW_F += - np.dot(np.exp(log_prob).T, xi)
            gW_T += - np.sum(np.exp(log_pairwise_marginals),axis=0)
            for j in range(seq_len):
                m1 = np.equal(self.label_to_idx[yi[j]],range(no_classes))
                gW_F += np.dot(m1[:,None], xi[j,:,None].T)
                
                if j<seq_len-1:
                    m2 = np.equal(self.label_to_idx[yi[j+1]],range(no_classes))
                    gW_T += np.dot(m1[:,None],m2[:,None].T)
            
        # your implementation here
        N = len(X)
        gW_F = gW_F/N
        gW_T = gW_T/N
        
        self.gW_F = gW_F
        self.gW_T = gW_T
        
        assert gW_T.shape == W_T.shape
        assert gW_F.shape == W_F.shape

        return (gW_F, gW_T)


    def fit(self, Y, X):
        '''
        Learns the CRF model parameters W_F, W_F given N labeled sequences as input.
        Sets the member variables W_T and W_F to the learned values

        Args:
            Y (list): a list of length N where each element n is a list of T_n labels from L
            X (list): a list of length N where each element n is a feature array of shape (T_n,F)

        Returns:
            None
        '''
        
        C, F = self.W_F.shape
        self.theta = fmin_l_bfgs_b(self.api_log_lhood, np.hstack((self.W_F,self.W_T)).flatten(), self.api_grad, args=(Y, X))[0].reshape((C,C+F))
        # self.theta = fmin_bfgs(self.api_log_lhood, np.hstack((self.W_F,self.W_T)).flatten(), self.api_grad, args=(Y, X)).reshape((C,C+F))
        self.W_F = self.theta[:,:F]
        self.W_T = self.theta[:,F:]
        
    def api_log_lhood(self, W, Y, X):
        C, F = self.W_F.shape
        W = W.reshape((C,C+F))
        W_F = W[:,:F]
        W_T = W[:,F:]
        return -1*self.log_likelihood(Y, X, W_F, W_T)
    
    def api_grad(self, W, Y, X):
        C, F = self.W_F.shape
        W = W.reshape((C,C+F))
        W_F = W[:,:F]
        W_T = W[:,F:]
        gW_F, gW_T = self.gradient_log_likelihood(Y, X, W_F, W_T)
        return -1*np.hstack((gW_F, gW_T)).flatten()

    def grad_check(self, Y, X, W_F=None, W_T=None):
        if W_F is None:
            W_F = self.W_F
        if W_T is None:
            W_T = self.W_T
        print check_grad(self.api_log_lhood, self.api_grad, np.hstack((W_F,W_T)).flatten(), Y, X)



"""
for c in range(gW_F.shape[0]):
    for f in range(gW_F.shape[1]):
        for j, yij in enumerate(yi):
            gW_F[c,f] += int(c==self.label_to_idx[yij])*xi[j,f] - xi[j,f]*np.exp(log_prob[j,c]) #log_prob dimension (T, |L|)

for c in range(gW_T.shape[0]):
        for cprime in range(gW_T.shape[1]):
        for j in range(len(yi)-1):
            yij = yi[j]
            yij_1 = yi[j+1]
            gW_T[c,cprime] += int(c==self.label_to_idx[yij])*int(cprime==self.label_to_idx[yij_1]) - np.exp(log_pairwise_marginals[j,c,cprime])
"""
"""
Author: Yousef Nami

Description:
------------
This file contains some of the functions that I created as extensions to some of the tutorials.
Specifically, it contains statistics functions.
"""
# libraries
import numpy as np

def gaussian(x, mu = 0, sigma = 1):
    """
    Gives the cumulative probability for each value in x based on a normal distribution defined by mu and sigma
    
    Dependencies:
    -------------
    import numpy as np
    
    Parameters:
    -----------
    
    x: np.array OR float
        this is the x values whose cumulative probability is to be found
    
    mu (optional -- defaults to 0): float
        this defines the mean of the normal distribution
    
    sigma (optional -- defaults to 1): float
        this defines the standard deviation of the normal distribution
        
    Returns:
    --------
    
    g: np.array
        np.array of the probabilities
    """
    
    x = np.asarray(x) if isinstance(x,list) else x
    
    g = 1/(np.sqrt(2*np.pi)* sigma) * np.exp(-1/2 * np.power((x-mu)/sigma,2))
    return g



def multi_gaussian(X, mu = 0, Sigma = 1):
    """
    Gives the cumulative probabilities for each value in X based on the normal distribution defined by mu
    and sigma
    
    Read the parameters as it contains important information about how the inputs are meant to be set up
    
    NOTE:
    -----
    Currently the function does not allow for setting custom correlations. Thus to set covariances, the user
    must input a covariance matrix.
    
    Dependencies:
    -------------
    import numpy as np
    
    
    Parameters:
    -----------
    
    X: np.array
        this is the array of the input data. The array is of shape n1 x n2, where n2 is the degrees of freedom
        of the array (i.e. the attributes). This defines the order of the multivariate problem. n1 refers to
        the number of datapoints
    
    mu (optional -- defaults to 0): list OR int
        depending on the input, it instantiates the mean vector. 
        - If 'int': creates a mean vector of constant mean
        - If 'list': sets the mean vector as the list
        
    Sigma (optional -- defaults to 1) list OR int OR np.array
        depending on the input, instantiates the covariance matrix.
        - If 'np.array': sets the covariance matrix Sigma as the input array
        - If 'int': creates a covariance matrix with all the standard deviations constant
        - If 'list': creates a covariance matrix with the standard deviations based on the values in 'list'
    
    Returns:
    --------
    p: list
        list of length n1. The probabilities for each datapoint
    """
    
    n = X.shape[1] # define the degrees of freedom
    data_dimensions = X.shape[0] # defines the number of datapoints
    
    mu = [mu for i in range(n)] if not mu else mu
    Sigma = (lambda x: x, lambda x: np.diag(x))[isinstance(Sigma, list)](
        (lambda x: x,lambda x: np.identity(n)*x)[isinstance(Sigma, int)](Sigma)
    )
    
    p = []
    
    for i in range(data_dimensions):
        p.append(
            1/((2*np.pi)**(n/2) * np.sqrt(np.linalg.det(Sigma))) * np.exp(
                -1/2 * np.matmul(np.matmul((X[i,:]-mu), np.linalg.inv(Sigma)), (X[i,:]-mu).T)
            )
        )
        
    # this was from an old iteration, what is does is put the correct probabilities in the diagonals
    # can you figure out what the off-diagonal terms mean?
    """p = 1/((2*np.pi)**(n/2) * np.sqrt(np.linalg.det(Sigma))) * np.exp(
        -1/2 * np.matmul(np.matmul(X-mu, np.linalg.inv(Sigma)), (X-mu).T)
    )"""
    
    return p



def covariance_matrix(X):
    """
    Calculates the covariance matrix for a given input X with n = X.shape[1] predictors
    
    Dependencies:
    -------------
    import numpy as np
    
    Parameters:
    -----------
    
    X: np.array
        the input array of the data. Array should be constructed in the following format: n1 x n2, where n1
        refers to the number of datapoints, and n2 refers to the dgrees of freedom (attributes)
    
    Returns:
    --------
    mu: list
        a list of the means of each attribute
    
    Sigma: np.array
        the covariance matrix for the inputs
    """
    n = X.shape[1]
    n_points = X.shape[0]
    Sigma = np.zeros([n,n])
    mu = [X[:,i].mean(axis = 0) for i in range(n)]
    for i in range(n): # rows
        for j in range(i,n):
            Sigma[i,j] = (X[:,i]*X[:,j]).sum(axis = 0)/n_points - mu[i]*mu[j]
            if i != j:
                Sigma[j,i] = Sigma[i,j]
            
    return mu,Sigma
"""A compilation of machine learning algorithms that I've built from scratch."""

import numpy as np

class LinearRegression:
    
    """
    Linear regression using gradient descent.
    """
    
    def __init__(self):
        
        self.theta=np.zeros(2)
        self.theta0_history=list()
        self.theta1_history=list()
        self.cost_history=list()
            
    
    def compute_cost(self, X, y, theta):
        """
        Compute cost for linear regression. Computes the cost of using theta as the
        parameter for linear regression to fit the data points in X and y.
        """
        
        m = y.size  
        cost = 0
        cost = np.sum(np.power((np.dot(X,theta) - y),2))/(2*m)
        
        return cost
    
    def gradient_descent(self, X, y, alpha, num_iters):
        """
        Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
        gradient steps with learning rate `alpha`.
        
        The first two thetas are stored for simple linear regression plots.
        """
        
        m = y.size
        theta = self.theta.copy()
        theta0_history = self.theta0_history.copy()
        theta1_history = self.theta1_history.copy()
        cost_history = self.cost_history.copy()
        for i in range(num_iters):
            dJdt = np.dot((np.dot(X,theta) - y).T, X)/m
            theta = theta - alpha*dJdt
            theta0_history.append(theta[0])
            theta1_history.append(theta[1])
            cost_history.append(self.compute_cost(X, y, theta))

        return theta, theta0_history, theta1_history, cost_history
    
    def fit(self, X, y, alpha=0.0001, num_iters=2000):
        
        """
        Fit the training data.
        
        Input:
        alpha: learning rate.
        num_iters: number of iterations.
        """
        
        # For multivariate linear regression.
        try:
            X=np.c_[np.ones(X.shape[0]), X]
            self.theta = np.zeros(X.shape[1])
        # For simple linear regression.
        except:
            X=np.stack([np.ones(X.size), X], axis=1)
        
        theta, theta0_history, theta1_history, cost_history = self.gradient_descent(X,y,alpha,num_iters)
        self.theta = theta
        self.theta0_history = theta0_history
        self.theta1_history = theta1_history
        self.cost_history = cost_history
        
    def predict(self,X):
        """
        Predict using calculated theta.
        """
        y_pred = np.dot(np.c_[np.ones(X.shape[0]), X], self.theta)
        
        return y_pred
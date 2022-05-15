import numpy as np
from my_algorithms import LogisticRegression
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score

# Set random seed.
np.random.seed(0)

# Load data.
data = np.loadtxt(os.path.join('data','logistic_regression_linear.txt'), delimiter=',')
X, y = data[:, 0:2], data[:, 2]

# Create instance of logistic regression and fit with chosen initial thetas.
lr = LogisticRegression()
lr.fit(X,y, alpha=0.0001, initial_theta=np.array([-24, 0.2, 0.2]))

# Create predictions with trained model.
y_pred = lr.predict(X)

# Calculate accuracy.
accuracy_score(y, y_pred)

# Retrieve the model parameters.
b = lr.theta[0]
w1, w2 = lr.theta[1], lr.theta[2]

# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2

# Plot the data, the classification, and the decision boundary.
xmin, xmax = 25, 105
ymin, ymax = 25, 105
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
plt.scatter(*X[y==0].T, s=8, alpha=0.5)
plt.scatter(*X[y==1].T, s=8, alpha=0.5)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel(r'$x_2$')
plt.xlabel(r'$x_1$')
plt.savefig('../output/logistic_regression/scatter_decision_boundary_linear.jpg', bbox_inches='tight')
plt.show()

# Load data.
data = np.loadtxt(os.path.join('data','logistic_regression_nonlinear.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]

# Create higher order features for non-linear decision boundary.
def map_features(X1, X2, degree=6):
    out = []
    for i in range(0, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))
    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)

# Convert features.
X = map_features(X[:,0],X[:,1])

# Create an instance of logistic regression and fit with L2-regularization.
lr = LogisticRegression()
lr.fit(X,y, alpha=0.01, L2=True)

# Create predictions with trained model.
y_pred = lr.predict(X)

# Calculate accuracy.
accuracy_score(y, y_pred)

# Plot cost against iterations.
plt.plot(np.arange(0,len(lr.cost_history)),lr.cost_history)
plt.savefig('../output/logistic_regression/plot_cost_iterations_nonlinear.jpg', bbox_inches='tight')
plt.show()
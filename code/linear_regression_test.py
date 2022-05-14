import numpy as np
from my_algorithms import LinearRegression
import matplotlib.pyplot as plt

# Set random seed.
np.random.seed(0)

# Generate linear data with noise for simple linear regression.
X = np.linspace(0,100,100)
delta = np.random.normal(-2,3,size=(100,))
y = .4*X + 3 + delta

# Create an instance of LinearRegression and fit training data.
lr = LinearRegression()
lr.fit(X,y)

# Create predictions with trained model.
y_pred = lr.predict(X)

# Plot actual against predicted values.
plt.scatter(y, y_pred)
plt.savefig('../output/linear_regression/scatter_actual_predicted_simple.jpg', bbox_inches='tight')
plt.show()

# Plot of data with best fit line.
plt.scatter(X,y)
plt.plot(X,lr.predict(X),"r")
plt.savefig('../output/linear_regression/scatter_best_fit.jpg', bbox_inches='tight')
plt.show()

# Plot of cost against iterations.
plt.plot(np.arange(0,len(lr.cost_history)),lr.cost_history)
plt.savefig('../output/linear_regression/plot_cost_iterations.jpg', bbox_inches='tight')
plt.show()

# Plot of cost against theta0.
plt.plot(lr.theta0_history,lr.cost_history)
plt.savefig('../output/linear_regression/plot_theta0_cost.jpg', bbox_inches='tight')
plt.show()

# Plot of cost against theta1.
plt.plot(lr.theta1_history,lr.cost_history)
plt.savefig('../output/linear_regression/plot_theta1_cost.jpg', bbox_inches='tight')
plt.show()

# 3D plot of thetas (x,y) and cost (z).
ax = plt.axes(projection='3d')
ax.scatter3D(lr.theta0_history, lr.theta1_history, lr.cost_history)
plt.savefig('../output/linear_regression/scatter3d_theta0_theta1_cost.jpg', bbox_inches='tight')
plt.show()

# Generate linear data with noise for multivariate linear regression.
X1 = np.linspace(0,100,100)
X2 = np.linspace(5,50,100)
X = np.c_[X1,X2]
delta = np.random.normal(-2,3,size=(100,))
y = -5*X1 + 3*X2 + 6 + delta

# Create an instance of LinearRegression and fit training data.
lr = LinearRegression()
lr.fit(X,y)

# Create predictions with trained model.
y_pred = lr.predict(X)

# Plot actual against predicted values.
plt.scatter(y, y_pred)
plt.savefig('../output/linear_regression/scatter_actual_predicted_multivariate.jpg', bbox_inches='tight')
plt.show()
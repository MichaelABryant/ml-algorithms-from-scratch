import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification


def compute_weighted_entropy_sum(split,y):
    
    split_1 = y[split]
    split_2 = y[np.invert(split)]
    
    if (len(split_1) == 0) or (len(split_2) == 0):
        return np.nan
    
    split_1_ratio = len(split_1)/len(split)
    split_2_ratio = len(split_2)/len(split)
    
    weighted_entropy_sum = split_1_ratio*compute_split_entropy(split_1) + split_2_ratio*compute_split_entropy(split_2)

    return weighted_entropy_sum

def compute_split_entropy(split):
    
    f = np.sum(split)/len(split)
    
    entropy_1 = 0
    entropy_2 = 0
    if f > 0:
        entropy_1 = -f*np.log(f)
    if (1-f) > 0:
        entropy_2 = -(1-f)*np.log(1-f)
        
    entropy = entropy_1 + entropy_2
    
    return entropy


X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=2)

sns.scatterplot(x=X[:,0], y=X[:,1], hue=y)

start_node_accuracy = np.sum(y)/len(y)

lowest_entropy = 1
lowest_entropy_feature = 0
lowest_entropy_threshold = 0
for feature in range(X.shape[1]):
    feature_min = min(X[:,feature])
    feature_max = max(X[:,feature])
    thresholds = np.linspace(feature_min,feature_max,1000)
    for i in thresholds:
        split = X[:,feature] > i
        entropy = compute_weighted_entropy_sum(split,y)
        if entropy < lowest_entropy:
            lowest_entropy = entropy
            lowest_entropy_feature = feature
            lowest_entropy_threshold = i
        
        
best_decision_boundary = np.array([lowest_entropy_threshold, lowest_entropy_threshold])
if lowest_entropy_feature == 0:
    span_decision_boundary = np.array([min(X[:,1]), max(X[:,1])])
    plt.plot(best_decision_boundary,span_decision_boundary, 'k', lw=0.5)
else:
    span_decision_boundary = np.array([min(X[:,0]), max(X[:,0])])
    plt.plot(span_decision_boundary,best_decision_boundary, 'k', lw=0.5)
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, legend=False)
plt.xlim(min(X[:,0]), max(X[:,0]))
plt.ylim(min(X[:,1]), max(X[:,1]))
plt.show()




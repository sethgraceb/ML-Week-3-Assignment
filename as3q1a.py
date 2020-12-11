#Q (i) a)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 5.0)
from sklearn.linear_model import LogisticRegression 
df = pd.read_csv("week3.csv", comment = '#')

#feature values
X1 = df.iloc[:, 0]  #1st feature on the x-axis
X2 = df.iloc[:, 1]  #2nd feature in the y-axis
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]   #target y on z-axis

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:, 0], X[:, 1], y, color = 'blue', label = 'training data')
ax.set_xlabel("Feature 1", fontsize = 15, color = 'blue')
ax.set_ylabel("Feature 2", fontsize = 15, color = 'blue')
ax.set_zlabel("Target value", fontsize = 15, color = 'blue')
plt.legend()

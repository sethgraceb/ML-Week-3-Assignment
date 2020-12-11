# i) (c)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 5.0)
df = pd.read_csv("week3.csv", comment = '#')

#feature values
X1 = df.iloc[:, 0]  #1st feature on the x-axis
X2 = df.iloc[:, 1]  #2nd feature in the y-axis
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]   #target y on z-axis

from sklearn.preprocessing import PolynomialFeatures
Xpoly = PolynomialFeatures(5).fit_transform(X)    #power of 5

mean_error = []; std_error = []
C = 1
from sklearn.linear_model import Lasso
model = Lasso(alpha = 1/(2 * C))
temp = []
model.fit(Xpoly, y)
ypred = model.predict(Xpoly)
from sklearn.metrics import mean_squared_error
intercept = model.intercept_
coef = model.coef_
mse = mean_squared_error(y, ypred)
#print("\n\nintercept:", intercept)   
print("coef:", coef)
#print("mean square error", mse)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(Xpoly[:, 0], Xpoly[:, 1], y, color = 'blue', label = 'training data')
ax.scatter(Xpoly[:, 0], Xpoly[:, 1], ypred, color = 'orange', label = 'predictions')
ax.set_xlabel("Feature 1", fontsize = 15, color = 'blue')
ax.set_ylabel("Feature 2", fontsize = 15, color = 'blue')
ax.set_zlabel("Target value", fontsize = 15, color = 'blue')
plt.legend()

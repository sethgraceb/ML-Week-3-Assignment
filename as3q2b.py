# ii b)

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
#print("X:", X)
#print("Xpoly", Xpoly)
mean_error = []; std_error = []
Ci_range = [0.5, 1, 5, 10, 50, 100]    #C values
for Ci in Ci_range:
    from sklearn.linear_model import Lasso
    model = Lasso(alpha = 1/(2 * Ci))
    temp = []
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 5)
    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train], y[train])
        ypred = model.predict(Xpoly[test])
        from sklearn.metrics import mean_squared_error
        intercept = model.intercept_
        coef = model.coef_
        mse = mean_squared_error(y[test], ypred)
        #print("\n\nintercept:", intercept)   
        #print("coef:", coef)
        print("mean square error:", mse)
        temp.append(mean_squared_error(y[test], ypred))
        #print("variance:", std_error)
    mean_error.append(np.array(temp).std())  #mean
    std_error.append(np.array(temp).std())   #variance

plt.errorbar(Ci_range, mean_error, yerr = std_error)
plt.title('Fold = 5')
plt.xlabel('Ci'); plt.ylabel('Mean Square Error')
#plt.xlim((0, 1000))

plt.legend(['Error Bar'])
plt.show()
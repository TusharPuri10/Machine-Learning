#train a logistic regression classifier to predict whether a flower is iris virginica or not
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
# print(list(iris.keys()))
# #['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']
print(iris['data'].shape)#rows column
# print(iris['target'])
# print(iris['DESCR'])
x = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)#true=1 false =0
# print(x)
# print👍


#train a logistic regression classifier
clf = LogisticRegression()
clf.fit(x, y)
exm = clf.predict(([[2.6]]))
# print(exm)
# using matplotlib to plot thw visualisation
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new, y_prob[:, 1], "g-", label="virginica")
plt.show()
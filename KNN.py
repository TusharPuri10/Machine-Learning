# Loading Modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading Dataset
iris = datasets.load_iris()

# Printing description and features
# print(iris.DESCR)
features = iris.data
labels = iris.target
# print(features[0], labels[0])

# Training the classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

preds = clf.predict([[5.1, 3.1, 1.4, 0.1]])

print(preds)

# importing all the modules
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# STEP-1 Building the Model !
#       Step-1.1 Gathering Data

# loading the dataset
diabetes = datasets.load_diabetes()

# Showing all the keys of the dataset
# print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# print(diabetes.DESCR) # Gives the description of dataset

#   step-1.2 Data Preparation

# if you want to use only 1 feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# # if you want to use all features
# diabetes_X = diabetes.data

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

#   step-1.3 Train Model
model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_y_train)

# STEP-2 Evaluating the Model ! (Train/Test Split)
diabetes_y_predicted = model.predict(diabetes_X_test)

print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))

print("weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# if you want to use only 1 feature then only graph can be plotted

plt.scatter(diabetes_X_test, diabetes_y_test)
plt.plot(diabetes_X_test, diabetes_y_predicted)
plt.show()

# Error if we predict using only 1 feature
# Mean squared error is:  3035.060115291269
# weights:  [941.43097333]
# Intercept:  153.39713623331644

# Error if we predict using all the features
# Mean squared error is:  1826.4841712795044
# weights:  [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067
#   458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ]
# Intercept:  153.05824267739402

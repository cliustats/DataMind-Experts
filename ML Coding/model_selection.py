# Model evaluation and and selection
 import numpy as np

 from sklearn.linear_model import LinearRegression
 from sklearn.preprocessing imort StandardScaler, PolynomialFeatures
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import mean_squared_error

 import tensorflow as tf

 import utils


 data = ...

 x = data[:, 0]
 y = data[;, 1]

 ## Split the dataset into training, cross validation, and test sets

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_ = train_test_split(x, y, test_size= 0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.5, random_state=1)

# Delete temporary variables
del x_, y_


## Fit a liner model
scaler_linear = StandardScaler()
X_train_scaled = scaler_linear.fit_transform(x_train)

## Train the model

linear_model = LinearRegression()

linear_model.fit(x_train_scaled, y_train)

## Evaluate the model
yhat = linear_model.predict(X_train_scaled)


# Use scikit-learn's utility function and divide by 2
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")


# for-loop implementation
total_squared_error = 0

for i in range(len(yhat)):
    squared_error_i = (yhat[i] - y_train[i])**2
    total_squared_error += squared_error_i

mse = total_squared_error / (2*len(yhat))


X_cv_scaled = scaler_linear.transform(x_cv)


# create the additional PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_mapped = poly.fit_transform(x_train)

scaled_poly = StandardScaler()

X_train_mapped_scaled = scaled_poly.fit_transform(X_train_mapped)



## adding polynomial features



# select the model with the lowest error
model_num = 3

# compute the test error
yhat = model_bc[model_num-1].predict(x_bc_test_scaled)
yhat = tf.math.sigmoid(yhat)
yhat = np.where(yhat >= threshold, 1, 0)
nn_test_error = np.mean(yhat != y_bc_test)

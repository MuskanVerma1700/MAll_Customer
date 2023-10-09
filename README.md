# MAll_Customer
Mall_Customer : Try to understand the sales in  peak hour and predict the hourly sales. using poly regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score #dataset
data = pd.read_csv('Mall_Customers.csv')
#relevant columns (Annual Income and Spending Score) X = data['Annual Income (k$)'].values.reshape(-1, 1)
y = data['Spending Score (1-100)'].values
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Perform polynomial regression
degree = 3
poly_features = PolynomialFeatures(degree=degree) X_train_poly = poly_features.fit_transform(X_train) X_test_poly = poly_features.transform(X_test) model = LinearRegression()
model.fit(X_train_poly, y_train)
#predictions
y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred) r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
# Visualization
plt.scatter(X, y, label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Polynomial Regression') plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
X = data['Annual Income (k$)'].values.reshape(-1, 1) y = data['Spending Score (1-100)'].values
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Perform polynomial regression
degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly, y_train)
#predictions
y_pred = model.predict(X_test_poly)
# Evaluate model
mse = mean_squared_error(y_test, y_pred) r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}') print(f'R-squared: {r2}')
# Visualization plt.figure(figsize=(12, 6))
# Training data
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, label='Training Data', color='blue')
X_train_sorted = np.sort(X_train, axis=0)
y_train_pred = model.predict(poly_features.transform(X_train_sorted)) plt.plot(X_train_sorted, y_train_pred, color='red', label='Fitted Polynomial Curve') plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Training Data and Fitted Polynomial Curve')
plt.legend()
# Testing data
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, label='Testing Data', color='blue')
X_test_sorted = np.sort(X_test, axis=0)
y_test_pred = model.predict(poly_features.transform(X_test_sorted)) plt.plot(X_test_sorted, y_test_pred, color='red', label='Fitted Polynomial Curve') plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Testing Data and Fitted Polynomial Curve')
plt.legend()
plt.tight_layout()
plt.show()

# relevant columns (Annual Income and Spending Score) X = data['Annual Income (k$)'].values.reshape(-1, 1)
y = data['Spending Score (1-100)'].values
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Perform polynomial regression
degree = 2
poly_features = PolynomialFeatures(degree=degree) X_train_poly = poly_features.fit_transform(X_train) X_test_poly = poly_features.transform(X_test) model = LinearRegression()
model.fit(X_train_poly, y_train)
# Make predictions
y_pred = model.predict(X_test_poly)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred) r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}') print(f'R-squared: {r2}')
# Visualization plt.figure(figsize=(12, 6))
# Scatter plot for actual spending scores

plt.scatter(X_test, y_test, label='Actual Spending Scores', color='blue')
# Sort X_test for plotting the fitted curve
X_test_sorted = np.sort(X_test, axis=0)
y_test_pred = model.predict(poly_features.transform(X_test_sorted))
# Plot the fitted polynomial curve
plt.plot(X_test_sorted, y_test_pred, color='red', label='Fitted Polynomial Curve') # Predicted spending scores for different annual incomes
X_range = np.arange(min(X_test), max(X_test), 1).reshape(-1, 1)
X_range_poly = poly_features.transform(X_range)
y_range_pred = model.predict(X_range_poly)
# comparison between actual and predicted spending scores
plt.plot(X_range, y_range_pred, color='green', linestyle='--', label='Predicted Spending Scores') plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Comparison of Actual and Predicted Spending Scores')
plt.legend()
plt.tight_layout()
plt.show()
# Define the business hours (10 AM to 8 PM) hours_per_day = 10
# Calculate hourly income based on annual income
data['Hourly Income (k$)'] = data['Annual Income (k$)'] / (hours_per_day * 365) # Assuming 365 days in a year
# Perform polynomial feature transformation

degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(data['Hourly Income (k$)'].values.reshape(-1, 1)) #trained model to predict hourly spending scores
hourly_spending_scores = model.predict(X_poly)
#estimated hourly spending scores to the dataset
data['Hourly Spending Score (1-100)'] = hourly_spending_scores
# first few rows of the transformed dataset
print(data[['Hourly Income (k$)', 'Hourly Spending Score (1-100)']].head()) # Define the business hours
hours_per_day = 10
# Calculate hourly income based on annual income
data['Hourly Income (k$)'] = data['Annual Income (k$)'] / (hours_per_day * 365) # Assuming 365 days in a year
# Perform polynomial feature transformation
degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(data['Hourly Income (k$)'].values.reshape(-1, 1)) # trained model to predict hourly spending scores
hourly_spending_scores = model.predict(X_poly)
#estimated hourly spending scores to the dataset
data['Hourly Spending Score (1-100)'] = hourly_spending_scores
#scatter plot to visualize the estimated hourly spending scores

plt.figure(figsize=(10, 6))
plt.scatter(data['Hourly Income (k$)'], data['Hourly Spending Score (1-100)'], label='Estimated Hourly Spending Scores', color='blue')
plt.xlabel('Hourly Income (k$)')
plt.ylabel('Hourly Spending Score (1-100)')
plt.title('Estimated Hourly Spending Scores vs. Hourly Income') plt.legend()
plt.grid(True)
plt.show()


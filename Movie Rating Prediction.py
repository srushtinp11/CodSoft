# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the IMDb Movies India dataset
data = pd.read_csv('IMDb_Movies_India.csv', encoding='ISO-8859-1')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Step 1: Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Strip spaces from column names (if any)
data.columns = data.columns.str.strip()

# Handle missing data: Drop rows with missing target values ('Revenue' and 'Rating')
if 'Revenue' in data.columns and 'Rating' in data.columns:
    data = data.dropna(subset=['Revenue', 'Rating'])
else:
    print("Columns 'Revenue' or 'Rating' are missing")

# Convert categorical features into numerical ones (if applicable)
# Example: Convert 'Genre' column to numerical categories using one-hot encoding
if 'Genre' in data.columns:
    data = pd.get_dummies(data, columns=['Genre'], drop_first=True)

# Feature selection (predict 'Revenue' using available features like 'Budget', 'Rating', 'Runtime')
# Check if the necessary columns exist in the dataset
if 'Budget' in data.columns and 'Rating' in data.columns and 'Runtime' in data.columns:
    X = data[['Budget', 'Rating', 'Runtime']]  # Independent variables
else:
    print("Columns 'Budget', 'Rating' or 'Runtime' are missing or named differently")

# Dependent variable (target variable)
if 'Revenue' in data.columns:
    y = data['Revenue']
else:
    print("Column 'Revenue' is missing")

# Step 2: Split the data into training and testing sets
if 'X' in locals() and 'y' in locals():  # Only split if X and y are defined
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Standardizing the data (optional, useful for certain models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Step 5: Make predictions using the trained model
    y_pred = model.predict(X_test_scaled)

    # Step 6: Evaluate the model performance
    print("Model Performance Evaluation:")

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse}")

    # R-squared (R2) Score
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared (R2) Score: {r2}")

    # Step 7: Visualization of the results
    # Plotting the real vs predicted sales (or revenue)
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, y_pred, color='red', label='Predicted Revenue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linestyle='--', label='Perfect Prediction')
    plt.title('Real vs Predicted Revenue')
    plt.xlabel('Real Revenue')
    plt.ylabel('Predicted Revenue')
    plt.legend()
    plt.show()

    # Step 8: Future Predictions (Optional)
    # Predict future revenue based on new input data
    new_movies = np.array([[10000000, 7.5, 120], [20000000, 8.0, 150]])  # Example new movies
    new_movies_scaled = scaler.transform(new_movies)
    future_revenue = model.predict(new_movies_scaled)

    print("Future Revenue Prediction for New Movies:")
    for movie, revenue in zip(new_movies, future_revenue):
        print(f"Budget: {movie[0]} | Rating: {movie[1]} | Runtime: {movie[2]} | Predicted Revenue: {revenue}")
else:
    print("Required columns (X and y) are not properly defined, check the dataset.")

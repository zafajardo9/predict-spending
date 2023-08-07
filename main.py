import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample historical transaction data 
data = {
    'timestamp': ['2023-08-01', '2023-08-02', '2023-08-03', '2023-08-04', '2023-08-05'],
    'amount': [100, 120, 80, 150, 200]
}

df = pd.DataFrame(data)

# Convert the timestamp to datetime and extract relevant features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# Prepare the features and target variable
X = df[['day_of_week', 'month']]
y = df['amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error as a measure of model performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Now, let's predict the spending for the next transaction (using the most recent data)
recent_data = {'day_of_week': [5], 'month': [8]}  # Assuming the user spends on a Saturday (day_of_week=5) in August (month=8)
next_transaction_prediction = model.predict(pd.DataFrame(recent_data))

print("Predicted amount for next transaction:", next_transaction_prediction[0])

# Calculate the probability of spending falling within a certain range
# For example, let's calculate the probability of spending between $90 and $110
error_std = np.sqrt(mse)  # Standard deviation of prediction errors
lower_bound = next_transaction_prediction[0] - error_std
upper_bound = next_transaction_prediction[0] + error_std

# Cumulative Distribution Function (CDF) of a normal distribution
probability_spending_range = (
    0.5 * (1 + np.math.erf((upper_bound - next_transaction_prediction[0]) / (error_std * np.sqrt(2))))
    - 0.5 * (1 + np.math.erf((lower_bound - next_transaction_prediction[0]) / (error_std * np.sqrt(2))))
)

print("Probability of spending between $90 and $110:", probability_spending_range)

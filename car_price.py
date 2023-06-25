import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 1: Load the dataset
df = pd.read_csv('car_data.csv')

# Step 2: Data Preprocessing
df = df.dropna()
df = pd.get_dummies(df, columns=['fueltype', 'aspiration', 'carbody', 'drivewheel'])
X = df.drop(['car_ID', 'CarName', 'price'], axis=1)
y = df['price']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Step 6: Save the trained model
joblib.dump(model, 'car_price_prediction_model.pkl')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/processed/featured_sales.csv")

print("Dataset Loaded Successfully!\n")

print(df.head())

print("\nDataset Shape:", df.shape)

# ==========================
# Select Features and Target
# ==========================

# Input features
X = df[["Year", "Month", "Quantity", "Discount"]]

# Target variable
y = df["Sales"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# ==========================
# Linear Regression Model
# ==========================

# Create the model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")

# ==========================
# Make Predictions
# ==========================

# Predict sales using test data
y_pred = lr_model.predict(X_test)

print("\nFirst 10 Predicted Sales:")
print(y_pred[:10])

print("\nFirst 10 Actual Sales:")
print(y_test.head(10).values)
# ==========================
# Model Evaluation
# ==========================

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n===== Model Evaluation =====")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
# ==========================
# Random Forest Model
# ==========================

# Create the model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluate the model
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = rf_mse ** 0.5
rf_r2 = r2_score(y_test, rf_pred)

print("\n===== Random Forest Evaluation =====")
print(f"Mean Absolute Error (MAE): {rf_mae:.2f}")
print(f"Mean Squared Error (MSE): {rf_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rf_rmse:.2f}")
print(f"R² Score: {rf_r2:.4f}")
# =====================================
# Future Sales Forecast
# =====================================

future_data = X.tail(12)

future_prediction = rf_model.predict(future_data)

print("\nFuture Sales Forecast:")

forecast = pd.DataFrame({
    "Predicted Sales": future_prediction
})

print(forecast)
# =====================================
# Forecast Visualization
# =====================================

import plotly.express as px

forecast["Month"] = range(1, len(forecast) + 1)

fig = px.line(
    forecast,
    x="Month",
    y="Predicted Sales",
    markers=True,
    title="Future Sales Forecast"
)

fig.update_layout(
    xaxis_title="Future Months",
    yaxis_title="Predicted Sales",
    template="plotly_white"
)

fig.write_html("images/sales_forecast.html")

print("Forecast chart saved successfully!")
# =====================================
# Save the Trained Model
# =====================================

joblib.dump(rf_model, "models/random_forest_model.pkl")

print("Model saved successfully!")
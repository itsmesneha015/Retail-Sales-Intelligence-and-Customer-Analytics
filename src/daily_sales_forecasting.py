import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("================================")
print("Daily Sales Forecasting")
print("================================")

# =================================
# Load Sales Dataset
# =================================

df = pd.read_csv(
    "data/raw/SampleSuperstore.csv",
    encoding="latin1"
)

print("\nSales dataset loaded successfully!")

# =================================
# Convert Order Date
# =================================

df["Order Date"] = pd.to_datetime(
    df["Order Date"]
)

# =================================
# Create Daily Sales Dataset
# =================================

daily_sales = (
    df.groupby("Order Date")["Sales"]
    .sum()
    .reset_index()
    .sort_values("Order Date")
)

daily_sales.columns = [
    "Date",
    "Sales"
]

# Fill missing dates with zero sales
date_range = pd.date_range(
    daily_sales["Date"].min(),
    daily_sales["Date"].max(),
    freq="D"
)

daily_sales = (
    daily_sales
    .set_index("Date")
    .reindex(date_range, fill_value=0)
    .rename_axis("Date")
    .reset_index()
)

# =================================
# Create Time Features
# =================================

daily_sales["Day"] = daily_sales["Date"].dt.day
daily_sales["Month"] = daily_sales["Date"].dt.month
daily_sales["Year"] = daily_sales["Date"].dt.year
daily_sales["DayOfWeek"] = daily_sales["Date"].dt.dayofweek

# =================================
# Create Lag Features
# =================================

daily_sales["Lag_1"] = daily_sales["Sales"].shift(1)
daily_sales["Lag_7"] = daily_sales["Sales"].shift(7)
daily_sales["Lag_14"] = daily_sales["Sales"].shift(14)

daily_sales["Rolling_7"] = (
    daily_sales["Sales"]
    .shift(1)
    .rolling(7)
    .mean()
)

# Remove rows with missing lag values
model_data = daily_sales.dropna().copy()

print("\n===== Daily Sales Dataset =====")
print(model_data.head())

# =================================
# Features and Target
# =================================

features = [
    "Day",
    "Month",
    "DayOfWeek",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Rolling_7"
]

X = model_data[features]
y = model_data["Sales"]

# =================================
# Time-Based Train-Test Split
# =================================

split_index = int(
    len(model_data) * 0.80
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# =================================
# Random Forest Model
# =================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print(
    "\nDaily Sales Forecasting Model "
    "trained successfully!"
)

# =================================
# Model Evaluation
# =================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n===== Model Evaluation =====")

print(
    f"Mean Absolute Error: {mae:.2f}"
)

print(
    f"R² Score: {r2:.4f}"
)

# =================================
# Prepare Future Dates
# =================================

future_days = 30

last_date = daily_sales["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=future_days,
    freq="D"
)

# =================================
# Recursive Future Forecast
# =================================

forecast_history = daily_sales[
    ["Date", "Sales"]
].copy()

future_predictions = []

for future_date in future_dates:

    lag_1_date = future_date - pd.Timedelta(days=1)
    lag_7_date = future_date - pd.Timedelta(days=7)
    lag_14_date = future_date - pd.Timedelta(days=14)

    lag_1 = forecast_history.loc[
        forecast_history["Date"] == lag_1_date,
        "Sales"
    ].iloc[0]

    lag_7 = forecast_history.loc[
        forecast_history["Date"] == lag_7_date,
        "Sales"
    ].iloc[0]

    lag_14 = forecast_history.loc[
        forecast_history["Date"] == lag_14_date,
        "Sales"
    ].iloc[0]

    rolling_values = forecast_history[
        (
            forecast_history["Date"] >=
            future_date - pd.Timedelta(days=7)
        )
        &
        (
            forecast_history["Date"] <
            future_date
        )
    ]["Sales"]

    rolling_7 = rolling_values.mean()

    future_input = pd.DataFrame([{
        "Day": future_date.day,
        "Month": future_date.month,
        "DayOfWeek": future_date.dayofweek,
        "Lag_1": lag_1,
        "Lag_7": lag_7,
        "Lag_14": lag_14,
        "Rolling_7": rolling_7
    }])

    prediction = model.predict(
        future_input
    )[0]

    prediction = max(
        0,
        prediction
    )

    future_predictions.append(
        prediction
    )

    forecast_history = pd.concat(
        [
            forecast_history,
            pd.DataFrame({
                "Date": [future_date],
                "Sales": [prediction]
            })
        ],
        ignore_index=True
    )

# =================================
# Create Forecast DataFrame
# =================================

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Sales": [
        round(value, 2)
        for value in future_predictions
    ]
})

# =================================
# Find Highest Predicted Sales Day
# =================================

highest_sales_day = forecast_df.loc[
    forecast_df["Predicted_Sales"].idxmax()
]

print(
    f"\n===== Predicted Daily Sales "
    f"for Next {future_days} Days ====="
)

print(
    forecast_df.to_string(index=False)
)

print(
    "\n===== Highest Expected Sales Day ====="
)

print(
    f"Date: "
    f"{highest_sales_day['Date'].date()}"
)

print(
    f"Predicted Sales: "
    f"{highest_sales_day['Predicted_Sales']:.2f}"
)

# =================================
# Save Forecast
# =================================

forecast_df.to_csv(
    "data/processed/daily_sales_predictions.csv",
    index=False
)

print(
    "\nDaily sales forecast saved successfully!"
)
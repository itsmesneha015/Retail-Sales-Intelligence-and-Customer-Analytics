import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("================================")
print("Daily Sales Forecasting V2")
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

# =================================
# Create Complete Date Range
# =================================

date_range = pd.date_range(
    daily_sales["Date"].min(),
    daily_sales["Date"].max(),
    freq="D"
)

daily_sales = (
    daily_sales
    .set_index("Date")
    .reindex(date_range)
    .fillna(0)
    .rename_axis("Date")
    .reset_index()
)

# =================================
# Time Features
# =================================

daily_sales["DayOfWeek"] = (
    daily_sales["Date"].dt.dayofweek
)

daily_sales["Day"] = (
    daily_sales["Date"].dt.day
)

daily_sales["Month"] = (
    daily_sales["Date"].dt.month
)

daily_sales["Year"] = (
    daily_sales["Date"].dt.year
)

daily_sales["WeekOfYear"] = (
    daily_sales["Date"].dt.isocalendar().week.astype(int)
)

# =================================
# Lag Features
# =================================

daily_sales["Lag_1"] = (
    daily_sales["Sales"].shift(1)
)

daily_sales["Lag_7"] = (
    daily_sales["Sales"].shift(7)
)

daily_sales["Lag_14"] = (
    daily_sales["Sales"].shift(14)
)

daily_sales["Lag_28"] = (
    daily_sales["Sales"].shift(28)
)

# =================================
# Rolling Features
# =================================

daily_sales["Rolling_7"] = (
    daily_sales["Sales"]
    .shift(1)
    .rolling(7)
    .mean()
)

daily_sales["Rolling_14"] = (
    daily_sales["Sales"]
    .shift(1)
    .rolling(14)
    .mean()
)

daily_sales["Rolling_28"] = (
    daily_sales["Sales"]
    .shift(1)
    .rolling(28)
    .mean()
)

# =================================
# Remove Missing Values
# =================================

model_data = daily_sales.dropna().copy()

print("\n===== Daily Forecasting Dataset =====")
print(model_data.head())

# =================================
# Features
# =================================

features = [
    "DayOfWeek",
    "Day",
    "Month",
    "Year",
    "WeekOfYear",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Lag_28",
    "Rolling_7",
    "Rolling_14",
    "Rolling_28"
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
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print(
    "\nImproved Daily Sales Model "
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
# Save Model Evaluation
# =================================

print("\n===== Evaluation Complete =====")

print(
    "The improved model will only be "
    "used for forecasting if its results "
    "are reasonable."
)
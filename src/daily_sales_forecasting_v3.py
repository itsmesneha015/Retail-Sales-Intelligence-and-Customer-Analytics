import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("================================")
print("Daily Sales Forecasting V3")
print("================================")

# =================================
# Load Dataset
# =================================

df = pd.read_csv(
    "data/raw/SampleSuperstore.csv",
    encoding="latin1"
)

print("\nSales dataset loaded successfully!")

# =================================
# Convert Date
# =================================

df["Order Date"] = pd.to_datetime(
    df["Order Date"]
)

# =================================
# Create Daily Sales
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
# Complete Date Range
# =================================

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
# Time Features
# =================================

daily_sales["DayOfWeek"] = (
    daily_sales["Date"].dt.dayofweek
)

daily_sales["DayOfMonth"] = (
    daily_sales["Date"].dt.day
)

daily_sales["Month"] = (
    daily_sales["Date"].dt.month
)

daily_sales["Year"] = (
    daily_sales["Date"].dt.year
)

daily_sales["WeekOfYear"] = (
    daily_sales["Date"]
    .dt.isocalendar()
    .week
    .astype(int)
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

daily_sales["Rolling_3"] = (
    daily_sales["Sales"]
    .shift(1)
    .rolling(3)
    .mean()
)

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

print("\n===== Forecasting Dataset =====")
print(model_data.head())

# =================================
# Features
# =================================

features = [
    "DayOfWeek",
    "DayOfMonth",
    "Month",
    "Year",
    "WeekOfYear",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Lag_28",
    "Rolling_3",
    "Rolling_7",
    "Rolling_14",
    "Rolling_28"
]

X = model_data[features]

y = model_data["Sales"]

# =================================
# Time-Based Split
# =================================

split_index = int(
    len(model_data) * 0.80
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# =================================
# Random Forest
# =================================

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print(
    "\nV3 Daily Sales Model "
    "trained successfully!"
)

# =================================
# Evaluation
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
# Baseline Comparison
# =================================

print("\n===== Model Comparison =====")

print("V1 → MAE: 1659.02 | R²: 0.0257")
print(
    f"V3 → MAE: {mae:.2f} | R²: {r2:.4f}"
)

print("\n===== Evaluation Complete =====")
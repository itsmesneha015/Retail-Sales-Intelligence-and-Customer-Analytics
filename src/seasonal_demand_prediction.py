import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("================================")
print("Seasonal Demand Prediction")
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
# Create Season
# =================================

def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4]:
        return "Spring"

    elif month in [5, 6]:
        return "Summer"

    elif month in [7, 8, 9]:
        return "Monsoon"

    else:
        return "Autumn"


df["Season"] = df[
    "Order Date"
].dt.month.apply(get_season)

# =================================
# Create Year
# =================================

df["Year"] = df[
    "Order Date"
].dt.year

# =================================
# Aggregate Seasonal Demand
# =================================

seasonal_demand = (
    df.groupby(
        ["Year", "Season", "Category"]
    )["Quantity"]
    .sum()
    .reset_index()
)

print("\n===== Seasonal Demand Dataset =====")

print(seasonal_demand.head())

# =================================
# Convert Categorical Columns
# =================================

model_data = pd.get_dummies(
    seasonal_demand,
    columns=["Season", "Category"]
)

# =================================
# Features and Target
# =================================

X = model_data.drop(
    columns=["Quantity"]
)

y = model_data["Quantity"]

# =================================
# Train-Test Split
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
    "\nSeasonal Demand Model "
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
# Prepare Future Data
# =================================

future_year = seasonal_demand[
    "Year"
].max() + 1

seasons = [
    "Winter",
    "Spring",
    "Summer",
    "Monsoon",
    "Autumn"
]

categories = [
    "Furniture",
    "Office Supplies",
    "Technology"
]

future_data = []

for season in seasons:

    for category in categories:

        future_data.append({
            "Year": future_year,
            "Season": season,
            "Category": category
        })

future_df = pd.DataFrame(
    future_data
)

# =================================
# Apply Same Encoding
# =================================

future_model_data = pd.get_dummies(
    future_df,
    columns=["Season", "Category"]
)

# Make sure future columns match training columns
future_model_data = future_model_data.reindex(
    columns=X.columns,
    fill_value=0
)

# =================================
# Predict Future Demand
# =================================

future_prediction = model.predict(
    future_model_data
)

future_df["Predicted_Quantity"] = (
    future_prediction.round(0).astype(int)
)

# =================================
# Display Predictions
# =================================

print(
    f"\n===== Predicted Seasonal Demand "
    f"for {future_year} ====="
)

print(
    future_df.to_string(index=False)
)

# =================================
# Find Top Category for Each Season
# =================================

top_future_categories = (
    future_df
    .sort_values(
        "Predicted_Quantity",
        ascending=False
    )
    .groupby("Season")
    .first()
    .reset_index()
)

print(
    "\n===== Expected High-Demand "
    "Category by Season ====="
)

print(
    top_future_categories.to_string(
        index=False
    )
)

# =================================
# Save Predictions
# =================================

future_df.to_csv(
    "data/processed/"
    "seasonal_demand_predictions.csv",
    index=False
)

top_future_categories.to_csv(
    "data/processed/"
    "future_top_category_by_season.csv",
    index=False
)

print(
    "\nSeasonal demand prediction files "
    "saved successfully!"
)
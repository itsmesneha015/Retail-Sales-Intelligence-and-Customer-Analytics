import pandas as pd

print("================================")
print("Seasonal Product Demand Analysis")
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

    elif month in [3, 4, 5]:
        return "Spring"

    elif month in [6, 7, 8]:
        return "Summer"

    else:
        return "Autumn"


df["Season"] = df[
    "Order Date"
].dt.month.apply(get_season)

# =================================
# Seasonal Sales Analysis
# =================================

seasonal_sales = (
    df.groupby("Season")["Sales"]
    .sum()
    .reset_index()
)

seasonal_sales = seasonal_sales.sort_values(
    "Sales",
    ascending=False
)

print("\n===== Sales by Season =====")

print(seasonal_sales)

# =================================
# Seasonal Quantity Analysis
# =================================

seasonal_quantity = (
    df.groupby("Season")["Quantity"]
    .sum()
    .reset_index()
)

seasonal_quantity = seasonal_quantity.sort_values(
    "Quantity",
    ascending=False
)

print("\n===== Quantity by Season =====")

print(seasonal_quantity)

# =================================
# Category Demand by Season
# =================================

category_season = (
    df.groupby(
        ["Season", "Category"]
    )["Sales"]
    .sum()
    .reset_index()
)

print("\n===== Category Sales by Season =====")

print(category_season)

# =================================
# Find Top Category for Each Season
# =================================

top_categories = (
    category_season
    .sort_values(
        ["Season", "Sales"],
        ascending=[True, False]
    )
    .groupby("Season")
    .first()
    .reset_index()
)

print("\n===== Top Category in Each Season =====")

print(top_categories)

# =================================
# Save Results
# =================================

seasonal_sales.to_csv(
    "data/processed/seasonal_sales.csv",
    index=False
)

seasonal_quantity.to_csv(
    "data/processed/seasonal_quantity.csv",
    index=False
)

category_season.to_csv(
    "data/processed/category_season_sales.csv",
    index=False
)

top_categories.to_csv(
    "data/processed/top_category_by_season.csv",
    index=False
)

print("\nSeasonal analysis files saved successfully!")
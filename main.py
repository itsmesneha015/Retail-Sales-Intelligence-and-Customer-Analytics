import pandas as pd
import numpy as np
import os
import plotly.express as px

# Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "raw", "SampleSuperstore.csv")
df = pd.read_csv(data_path, encoding="latin1")

# Display the first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Display the last 5 rows
print("\nLast 5 Rows:")
print(df.tail())

# Display dataset shape
print("\nShape of Dataset:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())
# ==========================
# Category-wise Sales
# ==========================

category_sales = (
    df.groupby("Category")["Sales"]
      .sum()
      .reset_index()
)

print("\nCategory Sales")
print(category_sales)

fig = px.bar(
    category_sales,
    x="Category",
    y="Sales",
    color="Category",
    title="Sales by Category",
    text_auto=".2s"
)

fig.update_layout(
    template="plotly_white"
)
fig.show()

print("Category Sales Chart Saved!")
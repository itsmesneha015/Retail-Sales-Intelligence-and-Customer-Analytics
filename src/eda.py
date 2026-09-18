import pandas as pd
import plotly.express as px

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_sales.csv")

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Create Month column
df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

# Monthly Sales
monthly_sales = df.groupby("Month")["Sales"].sum().reset_index()

fig = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    title="Monthly Sales Trend",
    markers=True
)

fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Total Sales",
    template="plotly_white"
)

fig.write_html("images/monthly_sales_trend.html")

print("Chart Saved Successfully!")

# ====================================
# CATEGORY SALES ANALYSIS
# ====================================

# Group sales by category
category_sales = df.groupby("Category")["Sales"].sum().reset_index()

print(category_sales)

# Create Bar Chart
fig = px.bar(
    category_sales,
    x="Category",
    y="Sales",
    color="Category",
    title="Category Sales Analysis",
    text_auto=".2s"
)

fig.update_layout(
    xaxis_title="Category",
    yaxis_title="Total Sales",
    template="plotly_white"
)

fig.write_html("images/category_sales.html")

print("Category Chart Saved Successfully!")

# ====================================
# REGION SALES ANALYSIS
# ====================================

# Group sales by region
region_sales = df.groupby("Region")["Sales"].sum().reset_index()

print("\nRegion Sales:")
print(region_sales)

# Create Region Sales Bar Chart
fig = px.bar(
    region_sales,
    x="Region",
    y="Sales",
    color="Region",
    title="Region Sales Analysis",
    text_auto=".2s"
)

# Improve chart appearance
fig.update_layout(
    xaxis_title="Region",
    yaxis_title="Total Sales",
    template="plotly_white"
)

# Save chart
fig.write_html("images/region_sales.html")

print("Region Chart Saved Successfully!")

# ====================================
# TOP 10 PRODUCTS ANALYSIS
# ====================================

# Calculate total sales for each product
top_products = (
    df.groupby("Product Name")["Sales"]
      .sum()
      .reset_index()
      .sort_values(by="Sales", ascending=False)
      .head(10)
)

print("\nTop 10 Products:")
print(top_products)

# Create Horizontal Bar Chart
fig = px.bar(
    top_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    color="Sales",
    title="Top 10 Products by Sales",
    text_auto=".2s"
)

# Highest sales on top
fig.update_layout(
    yaxis=dict(categoryorder="total ascending"),
    xaxis_title="Total Sales",
    yaxis_title="Product Name",
    template="plotly_white"
)

# Save chart
fig.write_html("images/top_products.html")

print("Top Products Chart Saved Successfully!")
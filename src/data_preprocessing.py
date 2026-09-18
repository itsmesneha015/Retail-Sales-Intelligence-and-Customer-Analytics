import pandas as pd

# ==========================
# Step 1: Load Dataset
# ==========================

df = pd.read_csv("data/raw/SampleSuperstore.csv", encoding="latin1")

print("=" * 50)
print("Dataset Loaded Successfully!")
print("=" * 50)

# ==========================
# Step 2: Check Missing Values
# ==========================

print("\nMissing Values:")
print(df.isnull().sum())
# ==========================
# Step 3: Check Duplicate Rows
# ==========================

print("\nDuplicate Rows:")
print(df.duplicated().sum())
# ==========================
# Step 4: Check Data Types
# ==========================

print("\nData Types:")
print(df.dtypes)
# ==========================
# Step 5: Convert Date Columns
# ==========================

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

print("\nUpdated Data Types:")
print(df.dtypes)
# ==========================
# Step 6: Remove Duplicate Rows
# ==========================

df.drop_duplicates(inplace=True)

print("\nShape After Removing Duplicates:")
print(df.shape)
# ==========================
# Step 7: Save Cleaned Dataset
# ==========================

df.to_csv("data/processed/cleaned_sales.csv", index=False)

print("\nCleaned dataset saved successfully!")
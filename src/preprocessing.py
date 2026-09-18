"""
preprocessing.py

Purpose:
--------
Clean the retail sales dataset and save the cleaned data.

Author: Sneha
Project: Retail Sales Analytics Dashboard
"""

import pandas as pd
import os

from data_loader import load_data


def preprocess_data(df):
    """
    Cleans the retail sales dataset.
    """

    print("\n========== DATA PREPROCESSING ==========\n")

    # -----------------------------
    # Dataset Shape
    # -----------------------------
    print("Original Shape:", df.shape)

    # -----------------------------
    # Missing Values
    # -----------------------------
    print("\nMissing Values:\n")
    print(df.isnull().sum())

    # Fill Postal Code if missing
    if "Postal Code" in df.columns:
        df["Postal Code"] = df["Postal Code"].fillna(0)

    # -----------------------------
    # Duplicate Rows
    # -----------------------------
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")

    df = df.drop_duplicates()

    # -----------------------------
    # Convert Date Columns
    # -----------------------------
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])

    # -----------------------------
    # Final Shape
    # -----------------------------
    print("\nFinal Shape:", df.shape)

    return df


if __name__ == "__main__":

    file_path = "data/raw/SampleSuperstore.csv"

    df = load_data(file_path)

    clean_df = preprocess_data(df)

    # Create processed folder if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)

    clean_df.to_csv(
        "data/processed/clean_sales.csv",
        index=False
    )

    print("\n✅ Clean dataset saved successfully!")
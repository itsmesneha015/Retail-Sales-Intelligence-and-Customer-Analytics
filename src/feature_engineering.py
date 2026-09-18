import pandas as pd
import numpy as np
import os


def create_features(df):
    """
    Create useful features from the cleaned dataset.
    """

    print("\n========== FEATURE ENGINEERING ==========\n")

    # Ensure Order Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["Order Date"]):
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    df = df.sort_values("Order Date").reset_index(drop=True)

    # ---------------------------------------
    # Date Features
    # ---------------------------------------
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.month_name()
    df["Quarter"] = df["Order Date"].dt.quarter
    df["Weekday"] = df["Order Date"].dt.day_name()

    # ---------------------------------------
    # Profit Margin
    # ---------------------------------------
    df["Profit Margin"] = np.where(
        df["Sales"] == 0,
        0,
        (df["Profit"] / df["Sales"]) * 100,
    )
    df["Profit Margin"] = df["Profit Margin"].replace(
        [np.inf, -np.inf], 0
    ).fillna(0)

    # ---------------------------------------
    # Sales Growth
    # ---------------------------------------
    df["Sales Growth"] = df["Sales"].pct_change().fillna(0)

    # ---------------------------------------
    # Rolling Average
    # ---------------------------------------
    df["Rolling Sales"] = (
        df["Sales"]
        .rolling(window=7, min_periods=1)
        .mean()
    )

    print("New Features Created Successfully!")

    return df
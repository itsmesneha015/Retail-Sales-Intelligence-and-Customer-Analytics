import pandas as pd

# =====================================
# Load Customer Entry/Exit Data
# =====================================

df = pd.read_csv(
    "data/processed/customer_entry_exit.csv"
)

print("Customer analytics data loaded successfully!")

# =====================================
# Basic Information
# =====================================

print("\n===== Dataset Information =====")

print("Total Frames:", len(df))

print(
    "Maximum Occupancy:",
    df["Current_Customers"].max()
)

print(
    "Average Occupancy:",
    round(
        df["Current_Customers"].mean(),
        2
    )
)

# =====================================
# Entry and Exit Statistics
# =====================================

total_entered = df["Total_Entered"].max()

total_exited = df["Total_Exited"].max()

print("\n===== Entry / Exit Analytics =====")

print("Total Customers Entered:", total_entered)

print("Total Customers Exited:", total_exited)

# =====================================
# Occupancy Analysis
# =====================================

average_occupancy = df[
    "Current_Customers"
].mean()

peak_occupancy = df[
    "Current_Customers"
].max()

minimum_occupancy = df[
    "Current_Customers"
].min()

print("\n===== Occupancy Analytics =====")

print(
    "Average Occupancy:",
    round(average_occupancy, 2)
)

print(
    "Peak Occupancy:",
    peak_occupancy
)

print(
    "Minimum Occupancy:",
    minimum_occupancy
)

# =====================================
# Business Summary
# =====================================

print("\n===== Business Summary =====")

print(
    f"Customers Entered: {total_entered}"
)

print(
    f"Customers Exited: {total_exited}"
)

print(
    f"Peak Store Occupancy: {peak_occupancy}"
)

print(
    f"Average Store Occupancy: "
    f"{average_occupancy:.2f}"
)
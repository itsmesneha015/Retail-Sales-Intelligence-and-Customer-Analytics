import pandas as pd

# Load customer detection data
df = pd.read_csv("data/processed/customer_analytics.csv")

print("\n===== Customer Analytics =====")

# Total number of frames analyzed
total_frames = len(df)

# Average number of customers
average_customers = df["Customer_Count"].mean()

# Maximum customers detected at one time
maximum_customers = df["Customer_Count"].max()

# Minimum customers detected
minimum_customers = df["Customer_Count"].min()

print("Total Frames Analyzed:", total_frames)
print("Average Customers:", round(average_customers, 2))
print("Maximum Customers:", maximum_customers)
print("Minimum Customers:", minimum_customers)
# Customer Traffic Over Frames

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(
    df["Frame"],
    df["Customer_Count"]
)

plt.xlabel("Frame Number")
plt.ylabel("Number of Customers")
plt.title("Customer Traffic Over Time")

plt.grid(True)

plt.savefig("images/customer_traffic.png")

plt.show()

print("Customer traffic graph saved successfully!")
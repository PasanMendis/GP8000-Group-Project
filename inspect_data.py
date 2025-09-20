import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the training data
df = pd.read_csv("train.csv")

# Make sure Gross is numeric
df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce")
df = df.dropna(subset=["Gross"])

# --- Histogram (raw values) ---
plt.figure(figsize=(8,5))
plt.hist(df["Gross"], bins=50, color="skyblue", edgecolor="black")
plt.xlabel("Gross ($)")
plt.ylabel("Count")
plt.title("Distribution of Gross (train.csv)")
plt.tight_layout()
plt.show()

# --- Histogram (log scale) ---
plt.figure(figsize=(8,5))
plt.hist(np.log1p(df["Gross"]), bins=50, color="salmon", edgecolor="black")
plt.xlabel("log(1 + Gross)")
plt.ylabel("Count")
plt.title("Distribution of log Gross (train.csv)")
plt.tight_layout()
plt.show()

# --- Summary statistics ---
print(df["Gross"].describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99]))
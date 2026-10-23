import pandas as pd

print("Loading California Housing data...")
df = pd.read_csv("housing.csv")

# We update the exact string names to match the physical CSV file's headers (snake_case).
columns_to_keep = [
    'median_house_value', # Our Target output
    'median_income',      # Input variable
    'housing_median_age', # Input variable
    'total_rooms',        # Input variable
    'population'          # Input variable
]

# This extracts only the numerical columns so the Statistical Engine does not crash on text.
df_clean = df[columns_to_keep]

# .dropna() deletes any rows with missing data (NaN) to prevent mathematical errors in the OLS statistics.
initial_rows = len(df_clean)
df_clean = df_clean.dropna()
print(f"Dropped {initial_rows - len(df_clean)} incomplete rows.")

# We extract 10,000 random rows to ensure the Streamlit web app maintains low latency during the presentation.
df_sample = df_clean.sample(n=10000, random_state=42)

# Save the pristine numerical dataset for the Auditor to process.
df_sample.to_csv("real_housing_data.csv", index=False)
print("Success! Created 'real_housing_data.csv' for the Auditor.")
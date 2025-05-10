import pandas as pd
import numpy as np

# Example: Replace this with your actual file path
file_path = "./dataa.csv"  # CSV file with the data shown above

# Set your desired even interval (e.g., 1.0 for hourly, 2.5 for every 2.5 hours)
desired_interval = 1.0  # in hours

# Load the data
df = pd.read_csv(file_path)

# Ensure the Time column is float
df['Time (Hr)'] = df['Time (Hr)'].astype(float)

# Set 'Time (Hr)' as the index
df.set_index('Time (Hr)', inplace=True)

# Create a new index with the desired even interval
new_index = np.arange(df.index.min(), df.index.max() + desired_interval, desired_interval)

# Reindex the dataframe to the new time index
df_resampled = df.reindex(new_index)

# Interpolate missing values linearly
df_interpolated = df_resampled.interpolate(method='linear')

# Reset index for output
df_interpolated = df_interpolated.reset_index().rename(columns={'index': 'Time (Hr)'})

# Optionally save to a new file
df_interpolated.to_csv("interpolated_output.csv", index=False)

print("Interpolation complete. Output saved to 'interpolated_output.csv'")


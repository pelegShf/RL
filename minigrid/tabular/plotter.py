import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files
df1 = pd.read_csv('episode_data_abs.csv')
df2 = pd.read_csv('episode_data_rel.csv')

# Plot the data from the first file
plt.plot(df1['Steps'], label='File 1')

# Plot the data from the second file
plt.plot(df2['Steps'], label='File 2')

# Add a legend
plt.legend()

# Show the plot
plt.show()
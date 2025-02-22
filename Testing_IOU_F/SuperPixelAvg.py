import pandas as pd

# Load the data from the Excel file
input_file = r"D:\downloadFiles\front_3\TestingVideo\PredictedImagesByMyModel\SuperPixelMethods_1\results.xlsx"
df = pd.read_excel(input_file)

# Check the first few rows to understand the structure of the data
print(df.head())

# Group by 'Method' and 'n_segments' and calculate the average for each group
# We will exclude non-numeric columns like 'Frame' and 'Method' from averaging
columns_to_average = df.columns[3:]  # Assuming numeric data starts from the 4th column

# Grouping by 'Method' and 'n_segments', then calculating the average for each group
grouped_averages = df.groupby(['methode', 'n_segments'])[columns_to_average].mean()

# Resetting index to make 'Method' and 'n_segments' regular columns
grouped_averages.reset_index(inplace=True)

# Save the results to a new Excel file
output_file = r"D:\downloadFiles\front_3\TestingVideo\PredictedImagesByMyModel\SuperPixelMethods_1\averages_by_method_and_segments.xlsx"
grouped_averages.to_excel(output_file, index=False)

print("Averages have been calculated and saved to:", output_file)

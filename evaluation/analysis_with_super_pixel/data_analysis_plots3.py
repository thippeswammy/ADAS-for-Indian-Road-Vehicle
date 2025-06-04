import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load data from the Excel file
input_file = r"D:\downloadFiles\front_3\TestingVideo\PredictedImagesByMyModel\SuperPixelMethods1\averages_by_method_and_segments.xlsx"
df = pd.read_excel(input_file)

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Set up the style for seaborn plots
sns.set(style="whitegrid")

# Define the columns for metrics to plot
metrics = ['IoU', 'Precision', 'Recall', 'F1-Score', 'TP', 'TN', 'FP', 'FN']

# Create a list of unique methods
methods = df['methode'].unique()
count = 0
df['Number of superpixel area'] = (df['Number of superpixel area'] * 5000).astype(int)

# Plot each metric across different methods and segments
for metric in metrics:
    # Group by 'methode' and 'Number of superpixel area' and calculate the mean of each metric
    grouped_data = df.groupby(['methode', 'Number of superpixel area'])[metric].mean().unstack().T
    # Create a new figure for each metric with a resolution of 1000x400
    plt.figure(figsize=(10, 4), dpi=100)  # 1000x400 pixels = 10x4 inches * 100 dpi

    # Plot the grouped data as a bar chart
    grouped_data.plot(kind='bar', figsize=(10, 4))

    # Set the title, x-axis, and y-axis labels
    plt.title(f'{metric} for Different Methods and Number of superpixel area')
    plt.xlabel('Number of superpixel area')
    plt.ylabel(f'{metric}')

    # Adjust legend position
    plt.legend(title='slic Method', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display the plot
    plt.tight_layout()
    # plt.show()
    count += 1
    plt.savefig(f"Image{count}.png", dpi=100)  # Ensure the resolution is maintained when saving

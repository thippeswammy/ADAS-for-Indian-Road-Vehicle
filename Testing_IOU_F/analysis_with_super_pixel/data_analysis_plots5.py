import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Function to normalize the TP, TN, FP, FN values
def normalize_columns(data, metrics):
    for metric in metrics:
        if metric in ['TP', 'TN', 'FP', 'FN']:
            max_value = data[metric].max()  # Get max value for normalization
            data[metric] = data[metric] / max_value  # Normalize by dividing by max value
    return data

# Function to create radar chart
def radar_chart(data, metrics, title, output_file, colors, dpi=300):
    # Number of metrics
    num_vars = len(metrics)

    # Set up the angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Add the first value of angles to close the chart
    angles += angles[:1]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi, subplot_kw=dict(polar=True))  # Increased figure size

    # Plot each method
    methods = data['methode'].unique()
    for i, method in enumerate(methods):
        method_data = data[data['methode'] == method]
        values = method_data[metrics].mean().values.flatten().tolist()
        values += values[:1]  # To close the circle

        # Plot each method with a different color
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.25)
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=method)  # Line for the data

    # Hide radial ticks
    ax.set_yticklabels([])

    # Set the labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontweight='bold', fontsize=14)

    # Add title with larger font size
    ax.set_title(title, size=18, fontweight='bold', color='blue', va='bottom')

    # Adjust the position of the legend and make it more readable
    ax.legend(title="Method", bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=12)

    # Adjust layout to ensure everything fits without overlap
    plt.tight_layout()

    # Save the plot with high resolution
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')  # Save with tight bounding box for better fit

# Load the Excel file into a Pandas DataFrame
input_file = r"D:\downloadFiles\front_3\TestingVideo\PredictedImagesByMyModel\SuperPixelMethods1\averages_by_method_and_segments.xlsx"
df = pd.read_excel(input_file)

# Display the first few rows of the dataframe to understand its structure
print(df.head())

# Normalize the TP, TN, FP, FN columns
metrics = ['IoU', 'Precision', 'Recall', 'F1-Score', 'TP', 'TN', 'FP', 'FN']
df = normalize_columns(df, metrics)

# Set the title for the radar chart
title = 'Radar Chart for Method Comparison'

# Choose the output file name for saving the radar chart
output_file = 'radar_chart_comparison.png'

# Define colors for different methods (you can add more colors if needed)
colors = ['red', 'blue', 'green', 'purple', 'orange']

# Call the function to create the radar chart
radar_chart(df, metrics, title, output_file, colors)

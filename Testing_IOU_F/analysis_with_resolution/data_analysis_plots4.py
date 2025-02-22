import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# File path to the .xlsx file
file_path = r'D:\downloadFiles\front_3\TestingVideo\PredictedImagesByMyModel\PredictedImages'

# Output directory for saving images
output_dir = f"{file_path}/analysis"
os.makedirs(output_dir, exist_ok=True)
file_path = f"{file_path}/results6_6 - Copy.xlsx"

# Load the data
data = pd.read_excel(file_path)

# Clean up the 'Resolution' column by stripping extra spaces
data['Resolution'] = data['Resolution'].str.strip()

# 1. Extract all unique resolutions
all_resolutions = data['Resolution'].unique()

# 2. Define metrics for plotting
metrics = ['Mean IoU', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity']


# metrics = ['Mean IoU', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity', 'Total TP', 'Total TN', 'Total FP',
#            'Total FN', 'Total Time (s)']

# 3. Create a Radar (Spider) plot for each resolution
def radar_chart(data, metrics, title, output_file, colors, dpi=300):
    # Number of metrics
    num_vars = len(metrics)

    # Set up the angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Add the first value of angles to close the chart
    angles += angles[:1]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi, subplot_kw=dict(polar=True))  # Increased figure size

    # Plot each resolution
    for i, resolution in enumerate(all_resolutions):
        resolution_data = data[data['Resolution'] == resolution].iloc[0]
        values = resolution_data[metrics].values.flatten().tolist()
        values += values[:1]  # To close the circle

        # Plot each resolution with a different color
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.25)
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=resolution)  # Line for the data

    # Hide radial ticks
    ax.set_yticklabels([])

    # Set the labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontweight='bold', fontsize=12)

    # Add title and legend
    ax.set_title(title, size=15, fontweight='bold', color='blue', va='bottom')
    ax.legend(title="Resolution", bbox_to_anchor=(1.1, 1), loc='upper left')

    # Save the plot with high resolution
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)  # Set high DPI for better resolution


# Define some colors for each resolution (you can customize this list)
colors = sns.color_palette("tab20", len(all_resolutions))  # Get a palette with enough colors

# Plot the radar chart for all resolutions on a single plot
radar_chart(data, metrics, "Radar Plot for All Resolutions", f"{output_dir}\\radar_plot_all_resolutions.png", colors,
            dpi=300)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
data = pd.read_excel(
    r'D:\downloadFiles\front_3\TestingVideo\PredictedImagesByMyModel\PredictedImages\by_results6_6_copy.xlsx')

# Clean and preprocess data
data = data.dropna()  # Remove rows with missing values

# Add a column for total pixels in each resolution
data['Total Pixels'] = data['Resolution'].apply(lambda x: int(x.split('x')[0]) * int(x.split('x')[1]))

# Sort data by 'Total Pixels' (small to large resolutions)
data = data.sort_values('Total Pixels').reset_index(drop=True)

# Find the resolution with the highest Mean IoU
best_resolution = data.loc[data['Mean IoU'].idxmax()]
print("Best Resolution:")
print(f"Resolution: {best_resolution['Resolution']}")
print(f"Mean IoU: {best_resolution['Mean IoU']:.4f}")

# Save the cleaned and sorted dataset for reference
data.to_csv('cleaned_and_sorted_resolution_data.csv', index=False)
print("Cleaned and sorted data has been saved to 'cleaned_and_sorted_resolution_data.csv'.")

# Plot Mean IoU vs Resolution (sorted by Total Pixels)
plt.figure(figsize=(12, 8))
sns.barplot(x='Resolution', y='Mean IoU', data=data, palette='viridis', order=data['Resolution'])
plt.title('Mean IoU for Resolutions (Small to Large)')
plt.xlabel('Resolution (Sorted by Total Pixels)')
plt.ylabel('Mean IoU')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('mean_iou_vs_resolution_sorted.png')
plt.show()

# Plot resource usage trends (CPU, RAM, GPU) vs Resolution (sorted by Total Pixels)
resource_metrics = ['Avg CPU (%)', 'Avg RAM (%)', 'Avg GPU (%)']
plt.figure(figsize=(12, 8))
for metric in resource_metrics:
    plt.plot(data['Resolution'], data[metric], marker='o', label=metric)

plt.title('Resource Usage Trends (Small to Large Resolutions)')
plt.xlabel('Resolution (Sorted by Total Pixels)')
plt.ylabel('Usage (%)')
plt.xticks(ticks=range(len(data['Resolution'])), labels=data['Resolution'], rotation=90)
plt.legend(title="Resource Metrics")
plt.grid(True)
plt.tight_layout()
plt.savefig('resource_usage_vs_resolution_sorted.png')
plt.show()

# Plot trade-off between Mean IoU and Total Time
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Total Time (s)', y='Mean IoU', hue='Resolution', data=data, palette='deep')
plt.title('Trade-off Between Total Time and Mean IoU')
plt.xlabel('Total Time (s)')
plt.ylabel('Mean IoU')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Resolution")
plt.tight_layout()
plt.savefig('time_vs_mean_iou_sorted.png')
plt.show()

# Save summary statistics
summary_stats = data.describe()
summary_stats.to_csv('summary_statistics.csv')
print("Summary statistics have been saved to 'summary_statistics.csv'.")

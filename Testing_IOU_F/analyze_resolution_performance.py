import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
# Replace 'resolution_data.csv' with your actual file name
data = pd.read_excel(
    r'D:\downloadFiles\front_3\TestingVideo\PredictedImagesByMyModel1\PredictedImages\by_results6_6_copy.xlsx')  # Assuming tab-separated data

# Clean and preprocess data (if required)
data = data.dropna()  # Remove rows with missing values

# Find the resolution with the highest Mean IoU
best_resolution = data.loc[data['Mean IoU'].idxmax()]
print("Best Resolution:")
print(f"Resolution: {best_resolution['Resolution']}")
print(f"Mean IoU: {best_resolution['Mean IoU']:.4f}")

# Save the cleaned dataset for reference
data.to_csv('cleaned_resolution_data.csv', index=False)
print("Cleaned data has been saved to 'cleaned_resolution_data.csv'.")

# Plot Mean IoU vs Resolution
plt.figure(figsize=(12, 8))
sns.barplot(x='Resolution', y='Mean IoU', data=data, palette='viridis', hue='Resolution', dodge=False, legend=False)
plt.title('Mean IoU for Different Resolutions')
plt.xlabel('Resolution')
plt.ylabel('Mean IoU')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('mean_iou_vs_resolution.png')
plt.show()

# Plot resource usage trends (CPU, RAM, GPU) vs Resolution
resource_metrics = ['Avg CPU (%)', 'Avg RAM (%)', 'Avg GPU (%)']
plt.figure(figsize=(12, 8))
for metric in resource_metrics:
    plt.plot(data['Resolution'], data[metric], marker='o', label=metric)

plt.title('Resource Usage Trends for Different Resolutions')
plt.xlabel('Resolution')
plt.ylabel('Usage (%)')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resource_usage_vs_resolution.png')
plt.show()

# Plot trade-off between Mean IoU and Total Time
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Total Time (s)', y='Mean IoU', hue='Resolution', data=data, palette='deep')
plt.title('Trade-off Between Total Time and Mean IoU')
plt.xlabel('Total Time (s)')
plt.ylabel('Mean IoU')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('time_vs_mean_iou.png')
plt.show()

# Save summary statistics
summary_stats = data.describe()
summary_stats.to_csv('summary_statistics.csv')
print("Summary statistics have been saved to 'summary_statistics.csv'.")

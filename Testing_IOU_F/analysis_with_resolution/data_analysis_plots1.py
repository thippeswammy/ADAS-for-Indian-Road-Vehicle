import os

import matplotlib.pyplot as plt
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

# Display the data to verify the structure
print("Cleaned Data:")
print(data.head())

# Exclude non-numeric columns for correlation calculations
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Set seaborn style
sns.set(style="whitegrid")

# 1. Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of All Metrics")
plt.tight_layout()
plt.savefig(f"{output_dir}\\correlation_heatmap.png", dpi=1000)
# plt.show()

# 2. Pair Plot for Relationships Between Metrics
sns.pairplot(data, hue="Resolution", palette="husl", diag_kind="kde", height=3)
plt.suptitle("Pair Plot for Metric Relationships", y=1.02)
plt.savefig(f"{output_dir}\\pairplot_metrics.png", dpi=1000)
# plt.show()

# 3. Distribution Plots for Selected Metrics
metrics = ['Mean IoU', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity']
for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[metric], kde=True, bins=15, color="skyblue")
    plt.title(f"Distribution of {metric}")
    plt.xlabel(metric)
    plt.xticks(rotation=90)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{output_dir}\\distribution_{metric.lower().replace(' ', '_')}.png", dpi=1000)
    # plt.show()

# 4. Box Plots for Metrics by Resolution
plt.figure(figsize=(12, 8))
sns.boxplot(data=data, x="Resolution", y="Mean IoU", palette="Set2")
plt.title("Box Plot of Mean IoU by Resolution")
plt.xlabel("Resolution")
plt.xticks(rotation=90)
plt.ylabel("Mean IoU")
plt.tight_layout()
plt.savefig(f"{output_dir}\\boxplot_mean_iou.png", dpi=1000)
# plt.show()

# 5. Heatmap for Metric Trends Across Resolutions
trend_metrics = data[['Resolution', 'Mean IoU', 'Precision', 'Recall', 'F1-Score', 'Accuracy']]
trend_data = trend_metrics.set_index('Resolution').transpose()
plt.figure(figsize=(12, 8))
sns.heatmap(trend_data, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
plt.title("Heatmap for Metrics Across Resolutions")
plt.tight_layout()
plt.savefig(f"{output_dir}\\metric_trends_heatmap.png", dpi=1000)
# plt.show()

# 6. Stacked Bar Chart for TP, FP, TN, FN
stack_data = data[['Resolution', 'Total TP', 'Total FP', 'Total TN', 'Total FN']]
stack_data.set_index('Resolution', inplace=True)
stack_data.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab10')
plt.title("Stacked Bar Chart of TP, FP, TN, FN")
plt.xlabel("Resolution")
plt.xticks(rotation=90)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_dir}\\stacked_bar_tp_fp_tn_fn.png", dpi=1000)
# plt.show()

# 7. Line Plots for Time and Metrics
plt.figure(figsize=(12, 8))
for metric in ['Mean IoU', 'Precision', 'Recall', 'F1-Score', 'Accuracy']:
    plt.plot(data['Resolution'], data[metric], marker='o', label=metric)
plt.title("Metric Trends by Resolution")
plt.xlabel("Resolution")
plt.xticks(rotation=90)
plt.ylabel("Value")
plt.legend(title="Metrics")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}\\lineplot_metric_trends.png", dpi=1000)
# plt.show()

plt.figure(figsize=(12, 8))
plt.plot(data['Resolution'], data['Total Time (s)'], marker='o', color="purple", label="Total Time (s)")
plt.title("Processing Time by Resolution")
plt.xlabel("Resolution")
plt.xticks(rotation=90)
plt.ylabel("Total Time (seconds)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}\\lineplot_total_time.png", dpi=1000)
# plt.show()

# 8. Scatter Plots for Key Relationships
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x="FPR", y="FNR", hue="Resolution", size="Total TP", sizes=(50, 300), palette="viridis")
plt.title("Scatter Plot of FPR vs FNR")
plt.xlabel("False Positive Rate (FPR)")
plt.xticks(rotation=90)
plt.ylabel("False Negative Rate (FNR)")
plt.legend(title="Resolution", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{output_dir}\\scatter_fpr_fnr.png", dpi=1000)
# plt.show()

# 9. Violin Plot for Metric Distributions
plt.figure(figsize=(12, 8))
sns.violinplot(data=data, x="Resolution", y="Recall", palette="muted")
plt.title("Violin Plot of Recall by Resolution")
plt.xlabel("Resolution")
plt.ylabel("Recall")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{output_dir}\\violinplot_recall.png", dpi=1000)
# plt.show()

# 10. Summary Statistics
summary_stats = data.describe()
# print("Summary Statistics:")
# print(summary_stats)
summary_stats.to_csv(f"{output_dir}\\summary_statistics.csv")

# 11. Best Mode Results
best_mode = data.loc[data['Mean IoU'].idxmax()]
print("\nBest Mode Results:")
print(best_mode)

# Save the best mode details
best_mode.to_frame().to_csv(f"{output_dir}\\best_mode_results.csv")

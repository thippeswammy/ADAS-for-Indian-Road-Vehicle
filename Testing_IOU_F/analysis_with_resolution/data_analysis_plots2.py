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

# 1. Correlation Bar Plot (replace heatmap with bar plot)
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_data.corr()
correlation_matrix_unstacked = correlation_matrix.unstack().reset_index(name="Correlation")
correlation_matrix_unstacked.columns = ['Metric 1', 'Metric 2', 'Correlation']
correlation_matrix_unstacked = correlation_matrix_unstacked[
    correlation_matrix_unstacked['Metric 1'] != correlation_matrix_unstacked['Metric 2']]
sns.barplot(x="Correlation", y="Metric 1", hue="Metric 2", data=correlation_matrix_unstacked, palette="coolwarm")
plt.title("Correlation Bar Plot of All Metrics")
plt.tight_layout()
plt.savefig(f"{output_dir}\\correlation_barplot.png", dpi=1000)

# 2. Pair Plot for Relationships Between Metrics (converted to bar plot for metrics)
metrics = ['Mean IoU', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity']
pairplot_data = data[metrics].mean().reset_index()
pairplot_data.columns = ['Metric', 'Value']
plt.figure(figsize=(10, 6))
sns.barplot(x="Metric", y="Value", data=pairplot_data, palette="husl")
plt.title("Average Metrics Bar Plot")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{output_dir}\\pairplot_metrics_bar.png", dpi=1000)

# 3. Distribution Plots for Selected Metrics (converted to bar plot)
for metric in metrics:
    plt.figure(figsize=(8, 6))
    value_counts = data[metric].value_counts().sort_index()
    sns.barplot(x=value_counts.index, y=value_counts.values, color="skyblue")
    plt.title(f"Distribution of {metric}")
    plt.xlabel(metric)
    plt.xticks(rotation=90)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{output_dir}\\distribution_{metric.lower().replace(' ', '_')}_bar.png", dpi=1000)

# 4. Box Plots for Metrics by Resolution (converted to bar plot)
plt.figure(figsize=(12, 8))
mean_iou_by_res = data.groupby('Resolution')['Mean IoU'].mean().reset_index()
sns.barplot(x='Resolution', y='Mean IoU', data=mean_iou_by_res, palette="Set2")
plt.title("Bar Plot of Mean IoU by Resolution")
plt.xticks(rotation=90)
plt.ylabel("Mean IoU")
plt.tight_layout()
plt.savefig(f"{output_dir}\\barplot_mean_iou.png", dpi=1000)

# 5. Heatmap for Metric Trends Across Resolutions (converted to bar plot)
trend_metrics = data[['Resolution', 'Mean IoU', 'Precision', 'Recall', 'F1-Score', 'Accuracy']]
trend_metrics_melted = trend_metrics.melt(id_vars='Resolution', var_name='Metric', value_name='Value')
plt.figure(figsize=(12, 8))
sns.barplot(x="Resolution", y="Value", hue="Metric", data=trend_metrics_melted, palette="Blues")
plt.title("Bar Plot for Metrics Across Resolutions")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{output_dir}\\metric_trends_barplot.png", dpi=1000)

# 6. Stacked Bar Chart for TP, FP, TN, FN (unchanged)
stack_data = data[['Resolution', 'Total TP', 'Total FP', 'Total TN', 'Total FN']]
stack_data.set_index('Resolution', inplace=True)
stack_data.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab10')
plt.title("Stacked Bar Chart of TP, FP, TN, FN")
plt.xlabel("Resolution")
plt.xticks(rotation=90)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_dir}\\stacked_bar_tp_fp_tn_fn.png", dpi=1000)

# 7. Line Plots for Time and Metrics (converted to bar plot)
metrics = ['Mean IoU', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
for metric in metrics:
    metric_time = data.groupby('Resolution')[metric].mean().reset_index()
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Resolution", y=metric, data=metric_time, palette="husl")
    plt.title(f"Bar Plot of {metric} by Resolution")
    plt.xticks(rotation=90)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"{output_dir}\\barplot_{metric.lower().replace(' ', '_')}_by_resolution.png", dpi=1000)

# 8. Scatter Plots for Key Relationships (converted to bar plot)
fpr_fnr_data = data[['Resolution', 'FPR', 'FNR']].groupby('Resolution').mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='Resolution', y='FPR', data=fpr_fnr_data, color='lightcoral', label="FPR")
sns.barplot(x='Resolution', y='FNR', data=fpr_fnr_data, color='mediumslateblue', label="FNR")
plt.title("Bar Plot of FPR and FNR by Resolution")
plt.xlabel("Resolution")
plt.ylabel("Rate")
plt.legend(title="Metrics")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{output_dir}\\barplot_fpr_fnr.png", dpi=1000)

# 9. Violin Plot for Metric Distributions (converted to bar plot)
plt.figure(figsize=(12, 8))
recall_by_res = data.groupby('Resolution')['Recall'].mean().reset_index()
sns.barplot(x='Resolution', y='Recall', data=recall_by_res, palette="muted")
plt.title("Bar Plot of Recall by Resolution")
plt.xlabel("Resolution")
plt.ylabel("Recall")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{output_dir}\\barplot_recall.png", dpi=1000)

# 10. Summary Statistics (converted to bar plot)
summary_stats = data.describe().transpose()
plt.figure(figsize=(10, 6))
summary_stats['mean'].plot(kind='bar', color="skyblue")
plt.title("Summary Statistics (Mean) of All Metrics")
plt.ylabel("Mean Value")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{output_dir}\\summary_statistics_bar.png", dpi=1000)

# 11. Best Mode Results (converted to bar plot)
best_mode = data.loc[data['Mean IoU'].idxmax()]
best_mode_df = best_mode.to_frame().transpose()
best_mode_df = best_mode_df.set_index('Resolution')
best_mode_df.plot(kind='bar', figsize=(12, 8), colormap='tab10')
plt.title("Best Mode Results (Highest Mean IoU)")
plt.xlabel("Resolution")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(f"{output_dir}\\best_mode_bar.png", dpi=1000)

# 12. Box Plots for Metrics by Resolution (converted to bar plot)
plt.figure(figsize=(12, 8))
mean_iou_by_res = data.groupby('Resolution')['Total Time (s)'].mean().reset_index()
sns.barplot(x='Resolution', y='Total Time (s)', data=mean_iou_by_res, palette="Set2")
plt.title("Bar Plot of Speed by Resolution")
plt.xticks(rotation=90)
plt.ylabel("Total Time (s)")
plt.tight_layout()
os.makedirs(output_dir, exist_ok=True)
# plt.savefig(os.path.join(output_dir, "barplot_Total_Time.png"), dpi=1000)

plt.savefig(f"{output_dir}\\barplot_Total_Time.png", dpi=1000)

# 13. Box Plots for Metrics by Resolution (converted to bar plot)
plt.figure(figsize=(12, 8))
mean_iou_by_res = data.groupby('Resolution')['Total TP'].mean().reset_index()
sns.barplot(x='Resolution', y='Total TP', data=mean_iou_by_res, palette="Set2")
plt.title("Bar Plot of Total TP by Resolution")
plt.xticks(rotation=90)
plt.ylabel("Total TP")
plt.tight_layout()
plt.savefig(f"{output_dir}\\barplot_Total_TP.png", dpi=1000)

# 14. Box Plots for Metrics by Resolution (converted to bar plot)
plt.figure(figsize=(12, 8))
mean_iou_by_res = data.groupby('Resolution')['Total TN'].mean().reset_index()
sns.barplot(x='Resolution', y='Total TN', data=mean_iou_by_res, palette="Set2")
plt.title("Bar Plot of Total TN by Resolution")
plt.xticks(rotation=90)
plt.ylabel("Total TN")
plt.tight_layout()
plt.savefig(f"{output_dir}\\barplot_Total_TN.png", dpi=1000)

# 15. Box Plots for Metrics by Resolution (converted to bar plot)
plt.figure(figsize=(12, 8))
mean_iou_by_res = data.groupby('Resolution')['Total FN'].mean().reset_index()
sns.barplot(x='Resolution', y='Total FN', data=mean_iou_by_res, palette="Set2")
plt.title("Bar Plot of Total TN by Resolution")
plt.xticks(rotation=90)
plt.ylabel("Total FN")
plt.tight_layout()
plt.savefig(f"{output_dir}\\barplot_Total_FN.png", dpi=1000)

# 16. Box Plots for Metrics by Resolution (converted to bar plot)
plt.figure(figsize=(12, 8))
mean_iou_by_res = data.groupby('Resolution')['Total FP'].mean().reset_index()
sns.barplot(x='Resolution', y='Total FP', data=mean_iou_by_res, palette="Set2")
plt.title("Bar Plot of Total FP by Resolution")
plt.xticks(rotation=90)
plt.ylabel("Total FP")
plt.tight_layout()
plt.savefig(f"{output_dir}\\barplot_Total_FP.png", dpi=1000)

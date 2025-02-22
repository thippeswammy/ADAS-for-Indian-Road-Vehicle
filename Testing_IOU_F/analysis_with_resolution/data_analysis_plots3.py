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

# 1. Correlation Density Plot (replace correlation bar plot with KDE plot)
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of All Metrics")
plt.tight_layout()
plt.savefig(f"{output_dir}\\correlation_heatmap.png", dpi=1000)

# 2. Pair Plot for Relationships Between Metrics (converted to density plot)
metrics = ['Mean IoU', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity']
pairplot_data = data[metrics].mean().reset_index()
pairplot_data.columns = ['Metric', 'Value']
plt.figure(figsize=(10, 6))
sns.kdeplot(data=pairplot_data['Value'], shade=True, color='skyblue', lw=2)
plt.title("Density Plot for Average Metrics")
plt.tight_layout()
plt.savefig(f"{output_dir}\\pairplot_metrics_density.png", dpi=1000)

# 3. Distribution Plots for Selected Metrics (replaced with density plots)
for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data[metric], shade=True, color="skyblue", lw=2)
    plt.title(f"Density Plot of {metric}")
    plt.xlabel(metric)
    plt.xticks(rotation=90)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"{output_dir}\\density_{metric.lower().replace(' ', '_')}.png", dpi=1000)

# 4. Box Plots for Metrics by Resolution (replaced with density plots)
plt.figure(figsize=(12, 8))
sns.kdeplot(data=data['Mean IoU'], shade=True, color='lightcoral')
plt.title("Density Plot of Mean IoU")
plt.xlabel("Mean IoU")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(f"{output_dir}\\density_mean_iou.png", dpi=1000)

# 5. Heatmap for Metric Trends Across Resolutions (converted to density plots)
trend_metrics = data[['Resolution', 'Mean IoU', 'Precision', 'Recall', 'F1-Score', 'Accuracy']]
trend_metrics_melted = trend_metrics.melt(id_vars='Resolution', var_name='Metric', value_name='Value')
plt.figure(figsize=(12, 8))
sns.kdeplot(data=trend_metrics_melted['Value'], shade=True, color='skyblue', lw=2)
plt.title("Density Plot for Metrics Across Resolutions")
plt.tight_layout()
plt.savefig(f"{output_dir}\\metric_trends_density.png", dpi=1000)

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

# 7. Line Plots for Time and Metrics (converted to density plot)
metrics = ['Mean IoU', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
for metric in metrics:
    metric_time = data.groupby('Resolution')[metric].mean().reset_index()
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=metric_time[metric], shade=True, color="skyblue", lw=2)
    plt.title(f"Density Plot of {metric} by Resolution")
    plt.xticks(rotation=90)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"{output_dir}\\density_{metric.lower().replace(' ', '_')}_by_resolution.png", dpi=1000)

# 8. Scatter Plots for Key Relationships (converted to density plot)
fpr_fnr_data = data[['Resolution', 'FPR', 'FNR']].groupby('Resolution').mean().reset_index()
plt.figure(figsize=(8, 6))
sns.kdeplot(data=fpr_fnr_data['FPR'], shade=True, color='lightcoral', label="FPR", lw=2)
sns.kdeplot(data=fpr_fnr_data['FNR'], shade=True, color='mediumslateblue', label="FNR", lw=2)
plt.title("Density Plot of FPR and FNR by Resolution")
plt.xlabel("Resolution")
plt.ylabel("Density")
plt.legend(title="Metrics")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{output_dir}\\density_fpr_fnr.png", dpi=1000)

# 9. Violin Plot for Metric Distributions (replaced with density plot)
plt.figure(figsize=(12, 8))
sns.kdeplot(data=data['Recall'], shade=True, color='lightseagreen', lw=2)
plt.title("Density Plot of Recall")
plt.xlabel("Recall")
plt.ylabel("Density")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{output_dir}\\density_recall.png", dpi=1000)

# 10. Summary Statistics (converted to density plot)
summary_stats = data.describe().transpose()
plt.figure(figsize=(10, 6))
sns.kdeplot(data=summary_stats['mean'], shade=True, color='skyblue', lw=2)
plt.title("Density Plot of Summary Statistics (Mean) of All Metrics")
plt.ylabel("Density")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{output_dir}\\summary_statistics_density.png", dpi=1000)

# 11. Best Mode Results (converted to density plot)
best_mode = data.loc[data['Mean IoU'].idxmax()]
best_mode_df = best_mode.to_frame().transpose()
best_mode_df = best_mode_df.set_index('Resolution')
sns.kdeplot(data=best_mode_df['Mean IoU'], shade=True, color='lightcoral', lw=2)
plt.title("Density Plot of Best Mode Results (Highest Mean IoU)")
plt.xlabel("Mean IoU")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(f"{output_dir}\\best_mode_density.png", dpi=1000)

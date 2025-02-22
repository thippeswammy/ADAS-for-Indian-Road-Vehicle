import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define the base directory for saving images
val = r"D:\downloadFiles\front_3\TestingVideo\PredictedImagesByMyModel\SuperPixelMethods1\analysis1"

# Load the data (assuming it's in Excel format)
data = pd.read_excel(r"D:\downloadFiles\front_3\TestingVideo\PredictedImagesByMyModel\SuperPixelMethods1\results.xlsx")

data['Number of superpixel area'] = (data['Number of superpixel area'] * 5000).astype(int)
# 1. IoU vs. Number of superpixel area
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Number of superpixel area', y='IoU', hue='methode', marker='o')
plt.title("IoU vs. Number of superpixel area for Different Methods")
plt.xlabel('Number of superpixel area')
plt.ylabel('IoU')
plt.grid(True)
plt.legend(title='Method')
plt.tight_layout()
plt.savefig(f'{val}\\IoU_vs_Number of superpixel area.png', dpi=1000)
# plt.show()


# 2. Precision vs. Recall
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Precision', y='Recall', hue='methode', style='methode')
plt.title("Precision vs. Recall for Different Methods")
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.grid(True)
plt.legend(title='Method')
plt.tight_layout()
plt.savefig(f'{val}\\Precision_vs_Recall.png', dpi=1000)
# plt.show()


# 3. F1-Score vs. IoU
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='IoU', y='F1-Score', hue='methode', marker='o')
plt.title("F1-Score vs. IoU for Different Methods")
plt.xlabel('IoU')
plt.ylabel('F1-Score')
plt.grid(True)
plt.legend(title='Method')
plt.tight_layout()
plt.savefig(f'{val}\\F1_Score_vs_IoU.png', dpi=1000)
# plt.show()


# 4. True Positives (TP), False Positives (FP) vs. Number of superpixel area
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Number of superpixel area', y='TP', hue='methode', marker='o')  # Removed label='TP'
sns.lineplot(data=data, x='Number of superpixel area', y='FP', hue='methode', marker='x')  # Removed label='FP'
plt.title("TP and FP vs. Number of superpixel area for Different Methods")
plt.xlabel('Number of superpixel area')
plt.ylabel('Count')
plt.legend(title='Method')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{val}\\TP_and_FP_vs_Number of superpixel area.png', dpi=1000)
# plt.show()


# 5. TP, TN, FP, FN for each method with respect to Number of superpixel area
melted_data = data.melt(id_vars=["methode", "Number of superpixel area"], value_vars=["TP", "TN", "FP", "FN"],
                        var_name="Metric", value_name="Count")
plt.figure(figsize=(12, 8))
sns.lineplot(data=melted_data, x="Number of superpixel area", y="Count", hue="Metric", style="methode", markers=True)
plt.title("TP, TN, FP, FN for Each Method vs. Number of superpixel area")
plt.xlabel('Number of superpixel area')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{val}\\TP_TN_FP_FN_vs_Number of superpixel area.png', dpi=1000)
# plt.show()

# 6. Box Plot for IoU Across Methods and Number of superpixel area
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='Number of superpixel area', y='IoU', hue='methode')
plt.title("Box Plot for IoU Across Methods and Number of superpixel area")
plt.xlabel('Number of superpixel area')
plt.ylabel('IoU')
plt.legend(title='Method')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{val}\\Box_Plot_IoU_vs_Number of superpixel area.png', dpi=1000)


# plt.show()


# 7. Radar Plot for Method Comparison (Precision, Recall, F1-Score, IoU)
def radar_plot(data, categories, values, title, save_path, dpi=300):  # Added dpi argument
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]  # To close the circle
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='b', alpha=0.25)
    ax.plot(angles, values, color='b', linewidth=2)  # Method line
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    plt.title(title, size=14, color='b', weight='bold', y=1.1)
    plt.tight_layout()

    # Save the figure with dpi
    plt.savefig(save_path, dpi=dpi)
    # plt.show()


# Example for a specific method:
# Select data for a specific method (e.g., "any one pixel" method)
method_data = data[data['methode'] == 'any one pixel']

# Get the average values for each metric at a specific n_segment (e.g., 0.5)
metrics = ['Precision', 'Recall', 'F1-Score', 'IoU']
values = [method_data[method_data['Number of superpixel area'] == 0.5][metric].mean() for metric in metrics]

# Plot radar chart
radar_plot(data=method_data, categories=metrics, values=values,
           title="Performance Metrics for 'any one pixel' Method (Number of superpixel area=0.5)",
           save_path=f'{val}\\Radar_Plot_Any_One_Pixel.png', dpi=1000)

# 8. Correlation Heatmap for Metrics
metrics_data = data[['IoU', 'Precision', 'Recall', 'F1-Score', 'TP', 'TN', 'FP', 'FN']]
correlation_matrix = metrics_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap for Metrics")
plt.tight_layout()
plt.savefig(f'{val}\\Correlation_Heatmap.png', dpi=1000)
# plt.show()


# 9. CDF for IoU Across Methods
plt.figure(figsize=(12, 6))
for method in data['methode'].unique():
    method_data = data[data['methode'] == method]
    sns.ecdfplot(method_data['IoU'], label=method)
plt.title("CDF for IoU Across Methods")
plt.xlabel('IoU')
plt.ylabel('Cumulative Probability')
plt.legend(title='Method')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{val}\\CDF_IoU.png', dpi=1000)
# plt.show()


# 10. Comparison of Best Metrics (F1-Score, Precision, IoU, etc.)
best_metrics = data.groupby('methode')[['Precision', 'Recall', 'F1-Score', 'IoU']].max()
# Plotting the best values for each metric
best_metrics.plot(kind='bar', figsize=(12, 6))
plt.title("Comparison of Best Values for Metrics Across Methods")
plt.xlabel('Method')
plt.ylabel('Metric Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{val}\\Best_Metrics_Comparison.png', dpi=1000)
# plt.show()


# 11. Best Method Based on IoU (Ranked by Number of superpixel area)
best_methods = data.groupby(['Number of superpixel area', 'methode'])['IoU'].mean().reset_index()
# Sort by IoU to find the best methods
best_methods = best_methods.sort_values(by=['Number of superpixel area', 'IoU'], ascending=[True, False])
# Plot the top-performing methods for each Number of superpixel area
plt.figure(figsize=(12, 6))
sns.lineplot(data=best_methods, x='Number of superpixel area', y='IoU', hue='methode', marker='o')
plt.title("Best Method Based on IoU Across Number of superpixel area")
plt.xlabel('Number of superpixel area')
plt.ylabel('IoU')
plt.legend(title='Method')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{val}\\Best_Methods_by_IoU.png', dpi=1000)
# plt.show()

# 12. F1-Score vs. Number of superpixel area (For Best Threshold)
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Number of superpixel area', y='F1-Score', hue='methode', marker='o')
plt.title("F1-Score vs. Number of superpixel area for Optimal Performance")
plt.xlabel('Number of superpixel area')
plt.ylabel('F1-Score')
plt.legend(title='Method')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{val}\\F1_Score_vs_Number of superpixel area.png', dpi=1000)
# plt.show()


# Heatmap for average IoU by methode and Number of superpixel area
# Assuming you want to group by methode as well
grouped_data = data.groupby(['methode', 'Number of superpixel area'], as_index=False).agg({'IoU': 'mean'})
pivot_table = grouped_data.pivot(index="methode", columns="Number of superpixel area", values="IoU")
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, annot=True, cmap="Blues", fmt=".4f")
plt.title("Average IoU by methode and Number of superpixel area")
plt.xlabel("Number of superpixel area")
plt.ylabel("methode")
plt.tight_layout()
plt.savefig(f'{val}\\Heatmap_Avg_IoU.png', dpi=1000)
print(pivot_table)
# plt.show()

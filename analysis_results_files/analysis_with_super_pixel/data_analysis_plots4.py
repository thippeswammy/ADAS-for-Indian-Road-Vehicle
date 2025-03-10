from math import pi

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load data from the Excel file
input_file = r"D:\downloadFiles\front_3\TestingVideo\PredictedImagesByMyModel\SuperPixelMethods1_1\averages_by_method_and_segments.xlsx"

df = pd.read_excel(input_file)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Style settings for seaborn plots
sns.set(style="whitegrid")

# Scaling 'Number of superpixel area' for better interpretability
df['Number of superpixel area'] = (df['Number of superpixel area'] * 5000).astype(int)

# List of metrics to analyze
metrics = ['IoU', 'Precision', 'Recall', 'F1-Score', 'TP', 'TN', 'FP', 'FN']

# 1. Line Plot for Trends Across Methods
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='Number of superpixel area',
        y=metric,
        hue='methode',
        marker="o",
        linewidth=2
    )
    plt.title(f"{metric} Across Superpixel Areas and Methods", fontsize=14)
    plt.xlabel('Number of superpixel area', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"line_plot_{metric}.png", dpi=150)
    # plt.show()

# 2. Heatmap for Correlation Analysis
plt.figure(figsize=(10, 8))
correlation_matrix = df[metrics].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Matrix of Metrics", fontsize=14)
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150)
# plt.show()

# 3. Regression Analysis for Each Metric
for metric in metrics:
    sns.lmplot(
        data=df,
        x='Number of superpixel area',
        y=metric,
        hue="methode",
        height=5,
        aspect=1.5,
        scatter_kws={"s": 50},
        line_kws={"linewidth": 2}
    )
    plt.title(f"Regression Analysis for {metric}", fontsize=14)
    plt.savefig(f"regression_{metric}.png", dpi=150)
    # plt.show()

# 4. Box Plot for Metric Distribution by Method
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="methode", y=metric, palette="Set2")
    plt.title(f"Distribution of {metric} Across Methods", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"boxplot_{metric}.png", dpi=150)
    # plt.show()

# 5. Radar Charts for Method Comparison
methods = df['methode'].unique()
for method in methods:
    method_data = df[df['methode'] == method][metrics].mean()

    categories = list(metrics)
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    values = method_data.tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=method)
    ax.fill(angles, values, alpha=0.4)
    plt.xticks(angles[:-1], categories, fontsize=10)
    plt.title(f"Radar Chart for {method}", fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(f"radar_chart_{method}.png", dpi=150)
    # plt.show()

# 6. Pairplot for Comparing Metrics
sns.pairplot(
    df,
    vars=metrics,
    hue="methode",
    diag_kind="kde",
    palette="deep",
    markers=["o", "s", "D", "P"][:len(df['methode'].unique())]
)
plt.savefig("pairplot_analysis.png", dpi=150)
# plt.show()

# 7. Summary Statistics
summary_stats = df.groupby('methode')[metrics].agg(['mean', 'std', 'median'])
print("Summary Statistics:")
print(summary_stats)
summary_stats.to_excel("metric_summary.xlsx")

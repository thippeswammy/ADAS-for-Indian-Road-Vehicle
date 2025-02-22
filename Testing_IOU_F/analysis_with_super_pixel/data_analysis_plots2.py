import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define the base directory for saving images
val = r"D:\downloadFiles\front_3\TestingVideo\PredictedImagesByMyModel\SuperPixelMethods\analysis"

# Load the data (assuming it's in Excel format)
data = pd.read_excel(r"D:\downloadFiles\front_3\TestingVideo\PredictedImagesByMyModel\SuperPixelMethods\results.xlsx")

# 1. Bar Graph for TP, FP, TN, FN
metrics_to_plot = ['TP', 'FP', 'TN', 'FN']
plt.figure(figsize=(14, 7))
data_melted = data.melt(id_vars=['methode'], value_vars=metrics_to_plot,
                        var_name='Metric', value_name='Count')
sns.barplot(data=data_melted, x='Metric', y='Count', hue='methode')
plt.title('Comparison of TP, FP, TN, FN Across Methods')
plt.xlabel('Metric')
plt.ylabel('Count')
plt.legend(title='Method')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{val}\\BarGraph_TP_FP_TN_FN.png', dpi=1000)
plt.show()

# 2. Bar Graph for IoU
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='methode', y='IoU', ci=None)
plt.title('Comparison of IoU Across Methods')
plt.xlabel('Method')
plt.ylabel('IoU')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{val}\\BarGraph_IoU.png', dpi=1000)
plt.show()

# 3. Bar Graph for F1-Score
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='methode', y='F1-Score', ci=None)
plt.title('Comparison of F1-Score Across Methods')
plt.xlabel('Method')
plt.ylabel('F1-Score')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{val}\\BarGraph_F1_Score.png', dpi=1000)
plt.show()

# 4. Bar Graph for Precision
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='methode', y='Precision', ci=None)
plt.title('Comparison of Precision Across Methods')
plt.xlabel('Method')
plt.ylabel('Precision')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{val}\\BarGraph_Precision.png', dpi=1000)
plt.show()

# 5. Bar Graph for Recall
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='methode', y='Recall', ci=None)
plt.title('Comparison of Recall Across Methods')
plt.xlabel('Method')
plt.ylabel('Recall')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{val}\\BarGraph_Recall.png', dpi=1000)
plt.show()

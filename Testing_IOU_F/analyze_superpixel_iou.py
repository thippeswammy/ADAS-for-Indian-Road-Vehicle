import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the Excel data
data = pd.read_excel(r'D:\downloadFiles\front_3\TestingVideo\PredictedImagesByMyModel1\SuperPixelMethods\results.xlsx')

# Group data by 'User Input' and 'n_segments' and calculate average IoU
grouped_data = data.groupby(['User Input', 'n_segments'])['IoU'].mean().reset_index()
grouped_data.rename(columns={'IoU': 'Avg_IoU'}, inplace=True)

# Find the combination with the highest average IoU
best_combination = grouped_data.loc[grouped_data['Avg_IoU'].idxmax()]

# Print the results
print("Best Configuration:")
print(f"User Input: {best_combination['User Input']}")
print(f"n_segments: {best_combination['n_segments']}")
print(f"Average IoU: {best_combination['Avg_IoU']:.4f}")

# Save the grouped data to a new CSV file for reference
grouped_data.to_csv('grouped_data.csv', index=False)
print("Grouped data with average IoUs has been saved to 'grouped_data.csv'.")

# Plotting the data
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Heatmap for average IoU by User Input and n_segments
pivot_table = grouped_data.pivot(index="User Input", columns="n_segments", values="Avg_IoU")
sns.heatmap(pivot_table, annot=True, cmap="Blues", fmt=".4f")
plt.title("Average IoU by User Input and n_segments")
plt.xlabel("n_segments")
plt.ylabel("User Input")
plt.tight_layout()
plt.savefig('heatmap_avg_iou.png')
plt.show()

# Line plot for IoU trends
plt.figure(figsize=(12, 8))
for user_input in grouped_data['User Input'].unique():
    subset = grouped_data[grouped_data['User Input'] == user_input]
    plt.plot(subset['n_segments'], subset['Avg_IoU'], marker='o', label=f"User Input {user_input}")

plt.title("IoU Trends for Different User Inputs")
plt.xlabel("n_segments")
plt.ylabel("Average IoU")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lineplot_iou_trends.png')
plt.show()
# Define custom colors for the bars

custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
# Bar plot for average IoU
plt.figure(figsize=(12, 8))
sns.barplot(data=grouped_data, x='n_segments', y='Avg_IoU', hue='User Input', palette=custom_palette)
plt.title("Average IoU by n_segments and User Input")
plt.xlabel("n_segments")
plt.ylabel("Average IoU")
plt.legend(title="User Input")
plt.tight_layout()
plt.savefig('barplot_avg_iou.png')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the confusion matrix
conf_matrix = np.array([[57969198688, 5292245997],
                        [1103695586, 72890590929]])

# Calculate the percentages
conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap="Blues", cbar=False,
            xticklabels=['Road', 'Predicted Pos'],
            yticklabels=['Actual Neg', 'Actual Pos'])

# Add labels and title
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix in Percentages')

# Show the plot
plt.show()

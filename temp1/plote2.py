import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the confusion matrix
cm = np.array([[39544587881, 4458244440],
               [1246450531, 35488406748]])
print("Confusion Matrix:\n", cm)

# Visualize confusion matrix
fig1 = plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Road", "Background"],
            yticklabels=["Road", "Background"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for YOLOv8-Segment (Pixel-Level)")
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# Save the figure for later editing
with open("confusion_matrix.pickle", "wb") as f:
    pickle.dump(fig1, f)

# Calculate the percentages
cmp = cm / cm.sum(axis=1, keepdims=True) * 100
fig2 = plt.figure(figsize=(8, 6))
sns.heatmap(cmp, annot=True, fmt=".2f", cmap="Blues", cbar=True, xticklabels=["Road", "Background"],
            yticklabels=["Road", "Background"])

# Add labels and title
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix for YOLOv8-Segment (Pixel-Level) in Percentages")
plt.savefig("confusion_matrix_percentages.png", dpi=300)
plt.show()

# Save the figure for later editing
with open("confusion_matrix_percentages.pickle", "wb") as f:
    pickle.dump(fig2, f)

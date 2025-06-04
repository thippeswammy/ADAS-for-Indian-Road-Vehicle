import pandas as pd
import matplotlib.pyplot as plt

# Load the data (update file path as needed)
file_path = "J:\\RoadSegmentationForMyDataset7\\results.csv"  # Ensure correct path format

# Try different delimiters if needed
df = pd.read_csv(file_path, delimiter=",")  # Change to '\t' if needed

# Strip column names to remove whitespace
df.columns = df.columns.str.strip()

# Print columns to verify correct parsing
print("Columns in CSV:", df.columns.tolist())

# If 'epoch' is not found, check if the entire header is in a single column
if len(df.columns) == 1:
    print("CSV file might be incorrectly formatted. Trying with another delimiter...")
    df = pd.read_csv(file_path, delimiter="\t")  # Retry with tab delimiter
    df.columns = df.columns.str.strip()
    print("Updated Columns in CSV:", df.columns.tolist())

# Final check for 'epoch' column
if 'epoch' not in df.columns:
    raise KeyError("The column 'epoch' is missing. Check CSV delimiter or formatting.")

# Extract epoch numbers
epochs = df['epoch']

# Define loss and metric columns to plot
loss_columns = ['train/box_loss', 'val/box_loss',
                'train/seg_loss', 'val/seg_loss',
                'train/cls_loss', 'val/cls_loss',
                'train/dfl_loss', 'val/dfl_loss']

metric_columns = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                  'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']

# Plot training and validation losses
plt.figure(figsize=(12, 6))
for i in range(0, len(loss_columns), 2):
    plt.plot(epochs, df[loss_columns[i]], label=loss_columns[i], linestyle='dashed', marker='o')
    plt.plot(epochs, df[loss_columns[i + 1]], label=loss_columns[i + 1], linestyle='solid', marker='s')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid()
plt.show()

# Plot metrics
plt.figure(figsize=(12, 6))
for col in metric_columns:
    plt.plot(epochs, df[col], label=col, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.title('Model Metrics Over Epochs')
plt.legend()
plt.grid()
plt.show()

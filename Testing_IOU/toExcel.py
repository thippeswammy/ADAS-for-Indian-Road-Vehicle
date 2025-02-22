import pandas as pd
import re

# File paths
txt_file = r"F:\RunningProjects\YOLO_Model\val\Results\evaluation_results.txt"
excel_file = r"F:\RunningProjects\YOLO_Model\val\Results\evaluation_results.xlsx"

# Initialize lists to store the data
system_usage_data = []
evaluation_data = []

# Read the .txt file
with open(txt_file, "r") as file:
    lines = file.readlines()

# Regular expressions for extracting data
system_usage_pattern = re.compile(r"(CPU Usage|RAM Usage|GPU Usage):\s*(\d+\.\d+)%|\s*(\d+\.\d+)\s*MB")
evaluation_pattern = re.compile(r"Evaluating for resolution: (\d+x\d+)")
metrics_pattern = re.compile(r"(Mean IoU|Total True Positives|Total True Negatives|Total False Positives|Total False Negatives|Total Time|Images Processed):\s*(\d+\.\d+|\d+)")

# Extract system usage data
for line in lines:
    system_usage_match = system_usage_pattern.search(line)
    if system_usage_match:
        system_usage_data.append(system_usage_match.groups())

# Extract evaluation data
current_resolution = None
metrics = {}

for line in lines:
    eval_match = evaluation_pattern.search(line)
    if eval_match:
        if current_resolution:
            evaluation_data.append(metrics)
        current_resolution = eval_match.group(1)
        metrics = {"Resolution": current_resolution}

    metric_match = metrics_pattern.search(line)
    if metric_match:
        metric_name = metric_match.group(1)
        metric_value = metric_match.group(2)
        metrics[metric_name] = metric_value

# Append last set of metrics
if metrics:
    evaluation_data.append(metrics)

# Create dataframes
system_usage_df = pd.DataFrame(system_usage_data, columns=["Metric", "Value1", "Value2"])
evaluation_df = pd.DataFrame(evaluation_data)

# Write to Excel
with pd.ExcelWriter(excel_file) as writer:
    system_usage_df.to_excel(writer, sheet_name="System Usage", index=False)
    evaluation_df.to_excel(writer, sheet_name="Evaluation Metrics", index=False)

print(f"Excel file created successfully: {excel_file}")

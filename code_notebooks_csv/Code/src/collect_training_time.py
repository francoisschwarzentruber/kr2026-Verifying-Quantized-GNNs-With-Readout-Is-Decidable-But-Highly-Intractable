import pandas as pd
import re
import glob
import os
import torch

activation_function='softplus'    # Options: "relu", "relu6", "trrelu", "gelu", "sigmoid", "silu", "softplus", "elu".
folder_path = rf'logging/results_synthetic/acrgnn_{activation_function}'


# Pattern to match files like p1-0-0time.time
file_pattern = os.path.join(folder_path, "p[1-3]-0-0time.time")
all_files = glob.glob(file_pattern)

# Regex to extract the layer and time from the file content
line_pattern = re.compile(r"Time taken for (\d+) layers: ([\d\.]+) seconds")

# Regex to extract 'p1', 'p2', etc. from the filename
filename_pattern = re.compile(r"(p[1-3])\-0\-0time\.time")

data = []

for filepath in all_files:
    filename = os.path.basename(filepath)
    match = filename_pattern.match(filename)
    if not match:
        print(f"Skipping unrecognized filename: {filename}")
        continue

    key = match.group(1)

    with open(filepath, 'r') as file:
        for line in file:
            line_match = line_pattern.search(line)
            if line_match:
                layer = int(line_match.group(1))
                time_seconds = float(line_match.group(2))
                data.append({
                    'key': key,
                    'layers': layer,
                    'time_seconds': time_seconds
                })

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv(f'for_analysis/new_act_functions/results_synthetic/acrgnn_{activation_function}_training_time.csv', index=False)
print(f"Saved as acrgnn_{activation_function}_timing.csv")
import os
import re
import pandas as pd

# Directory where your .time files are located
# Options: "relu", "relu6", "trrelu", "gelu", "sigmoid", "silu", "softplus", "elu". Is "relu" by default
#activation_function = "relu" 
for activation_function in ["relu", "relu6", "trrelu", "gelu", "sigmoid", "silu", "softplus", "elu"]:
    directory = "./logging/results_ppi/acrgnn_" + activation_function

    # Prepare list to collect data
    layer_time_data = []

    # Loop through files
    for filename in os.listdir(directory):
        if filename.endswith(".time"):
            # Extract number of layers from filename using regex (e.g., L10)
            match = re.search(r"L(\d+)", filename)
            if match:
                num_layers = int(match.group(1))

                # Read file content
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        if "Time taken" in line:
                            
                            time_match = re.search(r"Time taken.*?: ([\d\.]+)", line)
                            if time_match:
                                training_time = float(time_match.group(1))
                                layer_time_data.append({"Layers": num_layers, "Time (s)": training_time})

    # Convert to DataFrame
    df = pd.DataFrame(layer_time_data)

    print("Layer| Time (s)  DataFrame:")
    print(df)
    file_path= "./for_analysis/new_act_functions/results_ppi/"
    df.to_csv(f"{file_path}/LayerTime_{activation_function}.csv", index=False)
    print("Saved LayerTime.csv")

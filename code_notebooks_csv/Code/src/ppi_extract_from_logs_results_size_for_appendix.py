import re
import pandas as pd


for activation_function in ["relu", "relu6", "trrelu", "gelu", "sigmoid", "silu", "softplus", "elu"]:
    log_file=f"./for_analysis/new_act_functions/results_ppi/acrgnn_{activation_function}/ppi_{activation_function}_results_size_for_appendix.log"
    # Regex pattern to extract data
    pattern = r"acrgnn-L(\d+)-h256:Original model: ([\d.]+), Quantized model: ([\d.]+)"

    # Store parsed data
    data = []

    # Read and extract
    with open(log_file, 'r') as f:
        for line in f:
            match = re.match(pattern, line.strip())
            if match:
                layer = int(match.group(1))
                original = float(match.group(2))
                quantized = float(match.group(3))
                data.append((layer, original, quantized))

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Layers', 'Original (MB)', 'Quantized (MB)'])

    # Show the table
    print(df)

    # Optional: Save to CSV
    df.to_csv(f'./for_analysis/new_act_functions/results_ppi/acrgnn_{activation_function}/ppi_{activation_function}_results_size_for_appendix.csv', index=False)

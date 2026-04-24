import re
import pandas as pd


for activation_function in ["relu", "relu6", "trrelu", "gelu", "sigmoid", "silu", "softplus", "elu"]:
    log_file=f"./for_analysis/new_act_functions/results_ppi/acrgnn_{activation_function}/ppi_{activation_function}_results_time_for_appendix.log"
    # Regex pattern to extract data
    pattern = r"acrgnn-L(\d+)-h256:Elapsed Time Train: ([\d.]+), dPTQ Elapsed Time Train: ([\d.]+), Elapsed Time Test: ([\d.]+), dPTQ Elapsed Time Test: ([\d.]+), Elapsed Time Val: ([\d.]+), dPTQ Elapsed Time Val: ([\d.]+)"
    # Store parsed data
    data = []

    # Read and extract
    with open(log_file, 'r') as f:
        for line in f:
            match = re.match(pattern, line.strip())
            if match:
                layer = int(match.group(1))
                el_time_train_original = float(match.group(2))
                el_time_train_quantized = float(match.group(3))
                el_time_test_original = float(match.group(4))
                el_time_test_quantized = float(match.group(5))
                el_time_val_original = float(match.group(6))
                el_time_val_quantized = float(match.group(7))
                data.append((layer, el_time_train_original, el_time_train_quantized, el_time_test_original, el_time_test_quantized, el_time_val_original, el_time_val_quantized))


    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Layers', 'Elapsed_Time_Train', 'Elapsed_Time_Train_dPTQ', 'Elapsed_Time_Test', 'Elapsed_Time_Test_dPTQ', 'Elapsed_Time_Val', 'Elapsed_Time_Val_dPTQ'])

    # Show the table
    print(df)

    # Optional: Save to CSV
    df.to_csv(f'./for_analysis/new_act_functions/results_ppi/acrgnn_{activation_function}/ppi_{activation_function}_results_time_for_appendix.csv', index=False)

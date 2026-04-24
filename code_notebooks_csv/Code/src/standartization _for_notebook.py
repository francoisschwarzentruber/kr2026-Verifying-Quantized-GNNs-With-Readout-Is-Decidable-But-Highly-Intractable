import pandas as pd
import os

# === CONFIGURATION ===
#input_csv = r"E:\NeurIPS\Supplement_materials\Code\src\for_analysis\results_synthetic\dymanic_quantized_results_size_time_pytorch_acrgnn_tr_relu.csv"
#output_csv = r"E:\NeurIPS\Supplement_materials\Code\src\for_analysis\results_synthetic\standart_dymanic_quantized_results_size_time_pytorch_acrgnn_tr_relu.csv"
activation_function = 'elu'  # Options: "relu", "relu6", "trrelu", "gelu", "sigmoid", "silu", "softplus", "elu".
input_csv = fr"for_analysis\new_act_functions\results_synthetic\dymanic_quantized_results_size_time_pytorch_acrgnn_{activation_function}.csv"
output_csv = fr"for_analysis\new_act_functions\results_synthetic\standart_dymanic_quantized_results_size_time_pytorch_acrgnn_{activation_function}.csv"

column_name = "model_name"

# === FUNCTION TO TRANSFORM PATHS ===
def transform_model_path(path):
    new_path = path.replace(
        f"./saved_models/results_synthetic/acrgnn_{activation_function}/",
        "saved_models/results/"
    )

    return new_path

# === MAIN SCRIPT ===
if __name__ == "__main__":
    df = pd.read_csv(input_csv)

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in {input_csv}")

    df["model_name"] = df[column_name].apply(transform_model_path)
    df.to_csv(output_csv, index=False)

    print(f"Transformed model paths saved to: {output_csv}")
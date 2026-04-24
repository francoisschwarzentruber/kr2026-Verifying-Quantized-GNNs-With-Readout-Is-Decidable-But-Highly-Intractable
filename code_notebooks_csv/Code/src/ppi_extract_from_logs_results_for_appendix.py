import re
import pandas as pd


for activation_function in ["relu", "relu6", "trrelu", "gelu", "sigmoid", "silu", "softplus", "elu"]:
    log_file=f"./for_analysis/new_act_functions/results_ppi/acrgnn_{activation_function}/ppi_{activation_function}_results_for_appendix.log"
    # Regex pattern to extract data
    pattern = re.compile(
    r"acrgnn-L(\d+)-h256:Train Loss: ([\d.e+-]+|inf), Train Acc: ([\d.e+-]+|inf), Elapsed Time Train: ([\d.e+-]+|inf), "
    r"Test Loss: ([\d.e+-]+|inf), Test Acc: ([\d.e+-]+|inf), Elapsed Time Test: ([\d.e+-]+|inf), "
    r"Val Loss: ([\d.e+-]+|inf), Val Acc: ([\d.e+-]+|inf), Elapsed Time VAl: ([\d.e+-]+|inf)"
    )

    
    # Store parsed data
    data = []

    # Read and extract
    with open(log_file, 'r') as f:
        for line in f:
            match = re.match(pattern, line.strip())
            if match:
                layer = int(match.group(1))
                train_loss_original = float(match.group(2))
                train_acc_original = float(match.group(3))
                el_time_train_original = float(match.group(4))
                test_loss_original = float(match.group(5))
                test_acc_original = float(match.group(6))
                el_time_test_original = float(match.group(7))
                val_loss_original = float(match.group(8))
                val_acc_original = float(match.group(9))
                el_time_val_original = float(match.group(10))
                data.append((layer, train_loss_original, train_acc_original, el_time_train_original,test_loss_original, test_acc_original, el_time_test_original,  val_loss_original, val_acc_original, el_time_val_original))


    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Layers', 'Train_Loss_Original', 'Train_Acc_Original', 'Elapsed_Time_Train_Original', 'Test_Loss_Original', 'Test_Acc_Original', 'Elapsed_Time_Test_Original', 'Val_Loss_Original', 'Val_Acc_Original', 'Elapsed_Time_Val_Original'])

    # Show the table
    print(df)

    # Optional: Save to CSV
    df.to_csv(f'./for_analysis/new_act_functions/results_ppi/acrgnn_{activation_function}/ppi_{activation_function}_results_for_appendix.csv', index=False)

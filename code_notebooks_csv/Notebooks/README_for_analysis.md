# Files
In this section we are listing the `.csv` files that we are using for the analysis.

# Synthetic Dataset for the 3 Formulas of $\text{FOC}_2$ Logic 

## dataset.zip
Here are stored the datasets for the FOC_2 formulas. Please unzip them on foder `Supplement_materials\Code\src\data\datasets`

## How to obtain files for the analysis
Here, we need to describe in detail, step-by-step, which files should be run and in what order to obtain the results for the synthetic dataset.

### Preparation
You need to be sure that in the path `data\datasets\{key}` (where key `p1` or `p2` or `p3` ) you have 3 `.txt` files:
- `train-random-erdos-5000-40-50.txt`
- `test-random-erdos-500-40-50.txt`
- `test-random-erdos-500-51-60.txt`

To save the results of the training, you need to create two folders: one for logging files (`.log`) and another for the models (`.pth`).

For the logging files: `logging\results_synthetic\acrgnn_nameOfActivationFunction`.

For the model files: `saved_models\results_synthetic\acrgnn_nameOfActivationFunction\key`.

### explore_synthetich_data.py
This script collects the data used to generate the table titled "Dataset Statistics Summary" (Appendix: Experimental Data and Further Analyses). It provides minimum, maximum, and average statistics for synthetic datasets in terms of the number of nodes and edges.


### Models
You need to run the `main.py` file, where you need to modify:
- ACTIVATION = "elu"
- If you train a model:
```
_args.quantize = None
save_model=f"{file_path}/saved_models/{extra_name}{key}/MODEL-{_net_class}-{enum}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-cl{comb_layers}-L{l}-H{h}.pth",
train_model=True, 
```
- If you load the model and want to apply dynamic Post-Training Quantization:
```
_args.quantize = True
load_model=f"{file_path}/saved_models/{extra_name}{key}/MODEL-{_net_class}-0-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-cl{comb_layers}-L{l}-H{h}.pth",
train_model=False,  
```

Of course, first of all, you need to train models, and after that, you'll obtain the data for further steps.

### Files for the analysis
From `main.py` after training you will get for each key,the  number of layers L andthe  activation function:
-  `.log` file with the  **Loss** and **Accuracy** of model
-  `.time` file with the information about the training time: `Time taken for # layers: n seconds`
-  `.pth` file with the model that we'll use for the quantization

### Collect information
- To better understand the synthetic dataset, we can run `explore_synthetic_data.py` as output, and you get the LaTeX table in the terminal.
- To collect the training time of all models with one activation function in one `.csv` file we need to run `collect_training_time.py`
- To collect information about **Loss**, **Accuracy**, and **Size** of all models with one activation function in one `.csv` file we need to run `collect_data_before_dPTQ.py`
- To collect information about **Loss**, **Accuracy**, **Latency*,* and **Size** of all models with one activation function after applying dynamic Post-training Quantization in one `.csv` file we need to run `main.py`
- To procide the code in the notebooks we need to modify the column in the .csv files that we obtained from the `main.py` after applying dPTQ. In this case we need to use `standartization_for_notebooks.py`.


We summarize the information in table
| .py  | .csv |
| ------------- | ------------- |
| `collect_training_time.py`  | `acrgnn_{activation_function}_training_time.csv` |
| `collect_data_before_dPTQ.py`  | `non_qua_output_acrgnn_{activation_function}.csv`  |
| `collect_model_size_before_and_after_dPTQ.py` | `model_sizes_original_dyn_qua_pytorch_acrgnn_{activation_function}.csv` |
| `main.py` (with appropriate flags) | `dymanic_quantized_results_size_time_pytorch_acrgnn_{ACTIVATION}.csv` |
| `standartization_for_notebooks.py`| `standart_dymanic_quantized_results_size_time_pytorch_acrgnn_{activation_function}.csv`|


# Real Dataset The protein-protein interaction networks (PPI)

## How to obtain files for the analysis
Here, we need to describe in detail, step-by-step, which files should be run and in what order to obtain the results for the synthetic dataset.

### explore_ppi.py
This script supports the creation of two tables related to the PPI benchmark dataset:
- "Dataset Summary. PPI Benchmark" (Appendix: Experimental Data and Further Analyses) `DatasetSummaryppibenchmark.csv`
This table describes general statistics of the PPI dataset, including: Number of graphs, Node feature dimensions, Label dimensions, Average node degree. 

- "Dataset Statistics Summary. PPI Benchmark" (Appendix: Experimental Data and Further Analyses) `DatasetStatisticsSummaryppibenchmark.csv`
This table provides detailed statistical values, including minimum, maximum, and average values for the number of nodes and edges across graphs in the dataset.


### Models
You need to run the `run_ppi.py` file, where you need to modify:
- ACTIVATION = "elu"

Of course, first of all, you need to train models, and after that, you'll obtain the data for further steps.

### Files for the analysis
From `run_ppi.py`, after training you will get for each key,the  number of layers L and the activation function:
-  `.log` file with the  **Loss** and **Accuracy** of model
-  `.time` file with the information about the training time: `Time taken for # layers: n seconds`
-  `.pth` file with the model that we'll use for the quantization

### Collect information
- To understand better, a real-world dataset, we can run `explore_ppi.py` as output, and you get in the terminal the LaTeX table.
- To collect the training time of all models with one activation function in one `.csv` file, we need to run `ppi_training_time.py`
- To collect information about **Loss**, **Accuracy**, and **Elapsed Time** of all models with one activation function in one `.csv` file we need to run `ppi_extract_from_logs_results_for_appendix.py`
- To collect information about **Loss**, **Accuracy**, and **Elapsed Time** of all models with one activation function after applying dynamic Post-training Quantization in one `.csv` file , we need to run `ppi_extract_from_logs_quantized_results_for_appendix.py`
- To collect information about the **Size**  of all models with one activation function before and after applying dynamic Post-training Quantization in one `.csv` file, we need to run `ppi_extract_from_logs_results_size_for_appendix.py`
- To collect information about **Elapsed Time**  of all models with one activation function before and after applying dynamic Post-training Quantization in one `.csv` fil,e we need to run `ppi_extract_from_logs_results_time_for_appendix.py`


We summarize the information in a table
| .py  | .csv |
| ------------- | ------------- |
| `ppi_training_time.py`  | `LayerTime_{activation_function}.csv` |
| `ppi_extract_from_logs_results_for_appendix.py`  | `ppi_{activation_function}_results_for_appendix.csv`  |
| `ppi_extract_from_logs_quantized_results_for_appendix.py` | `ppi_{activation_function}_quantized_results_for_appendix.csv` |
| `ppi_extract_from_logs_results_size_for_appendix.py`  | `ppi_{activation_function}_results_size_for_appendix.csv` |
| `ppi_extract_from_logs_results_time_for_appendix.py`| `ppi_{activation_function}_results_time_for_appendix.csv`|

## References

The following references are related to the GNN-logic framework and theoretical background for the experiments.

**Paper**: [The Logical Expressiveness of Graph Neural Networks](https://openreview.net/forum?id=r1lZ7AEKvB)  
**Code**: [GNN-logic GitHub Repository](https://github.com/juanpablos/GNN-logic.git)  

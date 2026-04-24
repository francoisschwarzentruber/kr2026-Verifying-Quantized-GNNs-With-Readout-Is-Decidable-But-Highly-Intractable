# Introdcution

This repository is a fork of the codebase from Barceló et al. [1], originally developed for the paper "The Logical Expressiveness of Graph Neural Networks" [2].

In our submission, "On the Complexity of Verifying Quantized GNNs with Readout", we introduce several modifications to support our experiments and research objectives:

- `requirements.txt` — Updated using generate_requirements.py to reflect the versions of libraries used in our environment.

- Experiments — Conducted using both the synthetic datasets and the PPI benchmark, consistent with the original setup in [2].

- `README.md` — Revised to document our contributions and code changes clearly.

- `run_ppi.py` — Modified to include time measurement for the training process, necessary for preparing models for Post-Training Dynamic Quantization (PTQ).

- `main.py — Refactored to automate training and evaluation of ACR-GNN models on synthetic datasets (p1, p2, p3) and to optionally apply PTQ using PyTorch [3,4].

- The following scripts were added to support data analysis, reproducibility, and experiment automation. Each is described in more detail in the extended `README.md` accompanying the experimental results:
    - `generate_requirements.py`
    - `explore_synthetich_data.py`
    - `explore_ppi.py`
    - `ppi_collect_traing_time.py`
    - `run_ppi_dptq.py`
    - `collect_data_before_dPTQ.py`
    - `collect_model_sizes_before_and_after_dPTQ.py`
    - `standardization_for_notebook.py`


## Install

Run `pip install -r requirements.txt` to install all dependencies.


## Generate synthetic graphs

The graphs used in the paper are in the zip file `datasets.zip`. Just unzip them to `src/data/datasets`. The script expects three folders inside `src/data/datasets` named `p1`, `p2`, and `p3`. Thiìese data we obtain from the code [1].


## Replicate synthetic results
We modified this section according to the theoretical part of the article.

Run the script in `src/main.py`. The results will be printed to the console and logged in `src/logging/results`. A single file will collect the last epoch for each experiment for each dataset.

Example: `p2-0-0-acrgnn-aggS-readS-combT-cl1-L2` means:

* `p2`: the FOC2 property.
* `acrgnn`: the network being benchmarked, in this case ACR-GNN.
* `aggS`: the aggregation used, S stands for the SUM.
* `readS`: the readout used,  S stands for the SUM.
* `combT`: the combine used, T stands for the SIMPLE. SIMPLE a ReLU function is used to apply the non-linearity. No activation function is used over the output.
* `cl1`: the number of layers in the MLP used to weight each component (h, agg, readout), refered as `A`, `B` and `C` in the paper [2], `V`, `A` and `R` in the code. If 0, no weighting is done. If 1, a Linear unit is used.
* `L2`: the number of layers of the GNN. 2 in this case.
* 
### Collect information
- To better understand the synthetic dataset, we can run `explore_synthetic_data.py` as output, and you get the LaTeX table in the terminal.
- To collect the training time of all models with one activation function in one `.csv` file we need to run `collect_training_time.py`
- To collect information about **Loss**, **Accuracy**, and **Size** of all models with one activation function in one `.csv` file we need to run `collect_data_before_dPTQ.py`
- To collect information about **Loss**, **Accuracy**, **Latency*,* and **Size** of all models with one activation function after applying dynamic Post-training Quantization in one `.csv` file we need to run `main.py`
- To procide the code in the notebooks we need to modify the column in the .csv files that we obtained from the `main.py` after applying dPTQ. In this case we need to use `standartization_for_notebooks.py`.


## Replicate PPI results

Run the script in `src/run_ppi.py`. The results will be printed to the console and logged in `src/logging/ppi`. A single file will collect the last epoch for each GNN combination.
A file with no extension will be created with the mean of 10 runs for each configuration and the standard deviation.

### Collect information
- To understand better, a real-world dataset, we can run `explore_ppi.py` as output, and you get in the terminal the LaTeX table.
- To collect the training time of all models with one activation function in one `.csv` file, we need to run `ppi_training_time.py`
- To collect information about **Loss**, **Accuracy**, and **Elapsed Time** of all models with one activation function in one `.csv` file we need to run `ppi_extract_from_logs_results_for_appendix.py`
- To collect information about **Loss**, **Accuracy**, and **Elapsed Time** of all models with one activation function after applying dynamic Post-training Quantization in one `.csv` file , we need to run `ppi_extract_from_logs_quantized_results_for_appendix.py`
- To collect information about the **Size**  of all models with one activation function before and after applying dynamic Post-training Quantization in one `.csv` file, we need to run `ppi_extract_from_logs_results_size_for_appendix.py`
- To collect information about **Elapsed Time**  of all models with one activation function before and after applying dynamic Post-training Quantization in one `.csv` fil,e we need to run `ppi_extract_from_logs_results_time_for_appendix.py`

## References
[1] Pablo Barceló, Egor V. Kostylev, Mikaël Monet, Jorge Pérez, Juan L. Reutter, and Juan Pablo Silva.  
**GNN-logic**, GitHub repository, 2021.  
Available at: [https://github.com/juanpablos/GNN-logic](https://github.com/juanpablos/GNN-logic)

[2] Pablo Barceló, Egor V. Kostylev, Mikaël Monet, Jorge Pérez, Juan L. Reutter, and Juan Pablo Silva.  
**The Logical Expressiveness of Graph Neural Networks**, 8th International Conference on Learning Representations (ICLR), 2020.  
Available at: [https://openreview.net/forum?id=r1lZ7AEKvB](https://openreview.net/forum?id=r1lZ7AEKvB)

[3] PyTorch Team.  
  **Dynamic Quantization Recipe**.  
  [https://docs.pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html](https://docs.pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html)

[4] PyTorch Documentation.  
  **Post-Training Dynamic Quantization**.  
  [https://pytorch.org/docs/stable/quantization.html#post-training-dynamic-quantization](https://pytorch.org/docs/stable/quantization.html#post-training-dynamic-quantization)


## License

This project is licensed under the [MIT License](LICENSE).
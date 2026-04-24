#  Verifying Quantized GNNs With Readout Is Decidable But Highly Intractable

This repository contains code, data, and notebooks for the article 'Verifying Quantized GNNs With Readout Is Decidable But Highly Intractable''.
It includes experiments, result analysis, and visualizations aimed at understanding how quantization affects model performance and efficiency.

## Requirements

To run this code successfully, please ensure that:

- You have the correct version of **Python**:  
   Required: **Python 3.11.9**

- All required libraries are installed with compatible versions. You can do this by running:


`pip install -r requirements.txt`

## Project Files Overview --- Notebooks
> [Note!] We  divided the experiments for the separate notebooks to have more clear structure.

### experiments-with-tensors.ipynb
This notebook provides experiments with tensors using symmetric and asymmetric quantization schemes. It illustrates also the differences between INT8 and QINT8 formats.

### acrgnn-syntheticdata-trainingtime.ipynb
This notebook presents an analysis of the Training Time of the original models. Original models in terms of article[2] with implementation[1].

Key components of this notebook include:
- Activation function under analysis
- Plot Trainig time of activation function for different formulas
- Calculate total Trainig time of activation function for different formulas
- Percentage of configurations in which each activation function achieves the minimum training time, aggregated over layers 1--10 and all tested hidden dimensions and learning rates.
- Percentage of configurations in which each activation function achieves the maximum training time, aggregated over layers 1--10 and all tested hidden dimensions and learning rates.

The data for these experiments are presented on input.zip. More detailed path: './input/for_analysis/results_synthetic/training_times/'


### acrgnn-synthetic-accuracy.ipynb
This notebook presents an analysis of the Accuraccy of the models before and after dynamic Post-Training Quantization. Original models in terms of article[2] with implementation[1].

Key components of this notebook include two sections "Analysis of ACRGNN with different activation functions" and  "Analysis of the dynamic PTQ on ACRGNN with different activation functions":

For each section we made following computations:
- Read datasets
- Hyperparameter selection: 
  - Find the best value of the learning rate based on the accuracy
  - Find the best value of the hidden dimension based on the best learning rate
- Accuracy across 10 layers for 10 Activation functions. Heatmap plot
- Calculation Generalization Ratio and Generalization Gap
- LaTex tables "Generalization performance of the #-layer ACR-GNN with different activation functions (A/F), reported as both Generalization Ratios (Test/Train) and Generalization Gaps (Train – Test accuracy)."



The data for these experiments are presented on input.zip. 

More detailed path: './input/for_analysis/results_synthetic/loss_accuracy/'

More detailed path: './input/for_analysis/results_synthetic/dptq_loss_accuracy/'


The data for these experiments are presented on input.zip.

### acrgnn-syntheticdata-modelsize.ipynb
This notebook presents an analysis of the Size of the models before and after dynamic Post-Training Quantization. Original models in terms of article[2] with implementation[1].


### activation-functions-ppi-data.ipynb
This notebook presents a comprehensive analysis of quantization techniques applied to Aggregate Combined Graph Neural Networks with Global Readout (ACRGNNs) for the Protein-Protein Interactions (PPI) benchmark.

Key components of this notebook include:
- Pre-Trained models: Evaluation of pre-trained GNN models.
- Post-Training Quantization (PTQ): Evaluation of dynamic quantization on pre-trained GNN models.
- Accuracy Metrics: Comparison of model performance before and after quantization.
- Visualization: Interactive plots illustrating quantization effects on accuracy and weight distributions.

The data for these experiments are presented on kaggle.zip. More detailed path: kaggle\input\results_ppi

### kaggle.zip
This .zip contains all `.csv` files for the analysis that can be done with `.ipynb`.

More detailed information about `.cvs` files can be found in the file `README_for_analysis.md`.

### Folder 'fake_quantization'
In this folder, we conduct experiments using fake quantization to analyze the influence of bit-width on model performance.

Fake quantization simulates low-precision arithmetic (e.g., 8-bit, 4-bit, 2-bit) while keeping tensors stored in floating-point format (FP32). Instead of converting weights to true integer representations, quantization effects are emulated by:

- Applying quantization scaling (scale) and zero-point (zero_point)
- Rounding values to the target discrete levels
- De-quantizing back to floating-point for forward computation
- This approach allows us to:
- Evaluate accuracy degradation across different bit-widths
- Analyze sensitivity of model components to precision reduction

More details presented in 'README.md' in that folder. 

## Project Files Overview --- Code.py
This work builds on the GNN-Logic framework introduced by Barceló et al. [1], and follows the theoretical foundations from their ICLR 2020 paper on the logical expressiveness of GNNs [2].

### requirements.txt
Requirements to run a code with right versions of the libraries. Make sure you are using the correct Python version to run this code with PyTorch. You need **Python 3.11.9**, as not all PyTorch versions support all Python versions.

### dataset.zip
Here is stored the datasets for the FOC_2 folmulas. Please unzip them on the folder `Supplement_materials\Code\src\data\datasets`

### run_ppi.py
In this file, we refer to the code of the Barcelò et al. [1,2]. Authors described this file in the next way: The results will be printed to console and logged in `src/logging/results_ppi`. A single file will collect the last epoch for each GNN combination.
A file with no extension will be created with the mean of 10 runs for each configuration and the standard deviation.

We add the time measurements of the training time. The results will be printed in the console as `Time taken for {l} layers: {running_time} seconds`. A table with all the results of the running time can be formed after the file `ppi_collect_training_time.py`.

We also provided the `README.md` file from the original code [1] inside `src` folder.

### run_ppi_dptq.py
This Python script takes the trained models generated by `run_ppi.py`, applies the dynamic Post-Traing Quantization (PTQ) [3,4], and collects the data for the Appendix.  
The models' results provided: Accuracy and Loss for the Train, Validation, and Test splits.  

All measurements follow the same procedure:
1. Load the pre-trained model.
2. Run train, validation, and test data through the model.
3. Record accuracy, loss, model size, and inference time.

### main.py
This script is based on and extends the original code by Barcelò et al. [1,2], with our own modifications to support dynamic quantization and batch evaluation of ACR-GNN models on synthetic datasets.

It is designed to generate models for datasets p1, p2, and p3, evaluate their performance, and optionally apply Post-Training Dynamic Quantization (PTQ) using PyTorch.

Key Features
-Loads pre-trained ACR-GNN models (FP32) from disk.
-Evaluates each model on Train, Test1, and Test2 splits.
-Optionally trains models from scratch if enabled.
-Applies PTQ to convert models to quantized versions (QINT8).
-Supports both ReLU and truncated ReLU activation functions.
-Uses torch.quantization.quantize_dynamic for quantization of nn.Linear layers.
-Evaluation results and quantization stats are printed to the console and saved to disk.
-Logs performance metrics and evaluation summaries.


## Technical specification of the laptop
The experiments presented in this project were conducted on two different platforms:

- **Local Laptop**:  For `.py` a Samsung Galaxy Book4 laptop with an Intel Core i7-150U processor, 16 GB RAM, and 1 TB SSD storage. 
- **Kaggle Cloud Platform**.  For `.ipynb` notebooks executed on Kaggle: NVIDIA Tesla P100 GPU (16 GB RAM). Hosted runtime with pre-configured environment


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
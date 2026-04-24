#  Verifying Quantized GNNs With Readout Is Decidable But Highly Intractable
This repository contains data and notebooks for the article 'Verifying Quantized GNNs With Readout Is Decidable But Highly Intractable'.
It includes result analysis and visualizations aimed at understanding how quantization to different numbers of bits affects model performance.

## Project Files Overview --- Notebooks
### `kaggle.zip`
This archive contains all `.csv` files used for the analysis performed in the Jupyter notebooks.

**Instructions:**
1. Unzip this folder in the same directory as the Notebooks.
2. Ensure that the data is accessible under the path:  
   `/kaggle/input`

### `fake-quantization-synthetic-data.ipynb`
This notebook contains the analysis for the **synthetic dataset**.  
It includes:
1. An interactive Plotly visualization (requires running the notebook to view).
2. The dataframes used to generate the analysis included in the Appendix.
3. LaTeX tables generated for inclusion in the paper.

### `fake-quantization-ppi-data.ipynb`
This notebook contains the analysis for the **PPI dataset**.  
It includes:
1. An interactive Plotly visualization (requires running the notebook to view).
2. The dataframes used to generate the analysis included in the Appendix.
3. LaTeX tables generated for inclusion in the paper.

## License

This project is licensed under the [MIT License](LICENSE).
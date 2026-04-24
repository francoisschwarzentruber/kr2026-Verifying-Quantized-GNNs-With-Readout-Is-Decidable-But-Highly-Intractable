import pandas as pd
import re
import glob
import os
import torch

keys = ['p1', 'p2', 'p3']
activation_function = 'elu'  # Options: "relu", "relu6", "trrelu", "gelu", "sigmoid", "silu", "softplus", "elu".
model_sizes = {}
for key in keys:
    model_dir = rf'.\saved_models\results_synthetic\acrgnn_{activation_function}\{key}'
    for filename in os.listdir(model_dir):
        if filename.endswith('.pth'):
            model_path = os.path.join(model_dir, filename)
            model = torch.load(model_path, map_location='cpu')
            size = os.path.getsize(model_path) / 1e6  # Convert to MB
            print(f'{filename}: {size:.2f} MB')
            filename = filename.replace('MODEL', key)
            filename = filename.replace('-acgnn-0', '-0-0-acgnn')
            filename = filename.replace('-acrgnn-0', '-0-0-acrgnn')
            filename = filename.replace('-acrgnn-single-0', '-0-0-acrgnn-single')
            filename = filename.replace('-H64.pth', '.log')
            filename = filename.replace('-H64-quantized.pth', '-quantized.log')
            model_sizes[filename] = size
print('model size',model_sizes)

df = pd.DataFrame(model_sizes.items(), columns=['file', 'size'])
#print(df)
df_quantized = df[df['file'].str.endswith('-quantized.log')].reset_index(drop=True)
df = df[~df['file'].str.endswith('-quantized.log')].reset_index(drop=True)
print(df_quantized)
print(df)
df_quantized['file'] = df_quantized['file'].str.replace('-quantized.log', '.log')
df=df.join(df_quantized.set_index('file'), on='file', rsuffix='_quantized')
df['size_diff'] = df['size'] - df['size_quantized']
print(df)
df.to_csv(f"for_analysis/new_act_functions/results_synthetic/model_sizes_original_dyn_qua_pytorch_acrgnn_{activation_function}.csv", index=False)
print('done')
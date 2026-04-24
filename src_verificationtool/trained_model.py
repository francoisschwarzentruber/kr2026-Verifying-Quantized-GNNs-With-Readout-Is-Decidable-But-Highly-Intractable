import os
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import re 
import sys
from pathlib import Path

import torch
from torch import unsqueeze

from ESBMCVerificationTask import ESBMCVerificationTask

def activation_function_mapping(act:str)->str: 
    '''
    Map the activation function name to the corresponding representation used in the Z3VerificationTask.
    '''
    mapping = {
        'relu': 'ReLU',
        'relu6': 'ReLU6',
        'trrelu': 'trReLU'
    }
    if act in mapping:
        return mapping[act]
    else:
        raise ValueError(f"Unsupported activation function: {act}")


def extract_parameters_from_name_of_model(selection):

    # Activation
    act_match = re.search(r'acrgnn_([^/]+)', selection)
    if not act_match:
        raise ValueError("Activation not found")
    act_value = activation_function_mapping(act_match.group(1))

    # Layers
    L_value = int(re.search(r'-L(\d+)', selection).group(1))

    # Hidden dimension
    H_value = int(re.search(r'-H(\d+)', selection).group(1))

    # Epoch
    epoch_mathch = re.search(r'-epoch(\d+)', selection)

    epoch_value = int(epoch_mathch.group(1)) if epoch_mathch else 0  # default if not found

    # Learning rate
    lr_match = re.search(r'-lr(\d+(?:\.\d+)?)', selection)
    lr_value = float(lr_match.group(1)) if lr_match else 0.01  # default if not found

    # Quantization bits (NEW LOGIC)
    bit_match = re.search(r'-(\d+)-quantized', selection)
    bit_value = int(bit_match.group(1)) if bit_match else 32

    return act_value, L_value, H_value, bit_value, epoch_value, lr_value


def apply_bn_inference(bn_state, prefix, eps=1e-5): # need to be added to layer into class Z3 and ESBMC
    """
    x: Tensor shaped [N, C] or [*, C] depending on your model (C is feature dim).
    bn_state: state_dict-like mapping
    """
    gamma = bn_state[f"{prefix}.weight"]        # gamma (learned scale)
    beta = bn_state[f"{prefix}.bias"]          # beta (learned shift)
    mu = bn_state[f"{prefix}.running_mean"] # mu (EMA mean)
    sigma = bn_state[f"{prefix}.running_var"]  # var (EMA variance) sigma**2

    # training=False forces use of running stats
    a= gamma / torch.sqrt(sigma + eps)
    c = beta - mu * a
    return a,c


def trained_model_fp32(in_filename, in_configurations):
    path_to_model_folder='./Model'
    additional_path='acrgnn_relu'  
    selection=f'{additional_path}/MODEL-acrgnn-0-aggS-readS-combT-cl1-L1-H2-epoch20-lr0.01.pth'
    
    selected_model = f"{path_to_model_folder}/{selection}"
    print(f"Loading model from {selected_model}")
    in_activation, in_n_layers, in_dimension_of_hidden_layers,in_bitvect, in_epoch_value, in_lr_value= extract_parameters_from_name_of_model(selection)
    
    # state is typically an OrderedDict of parameter tensors
    state = torch.load(selected_model, map_location="cpu")
    
    print("Model state dictionary:")
    print(type(state), len(state))
    print("keys:", list(state.keys()))

    path = r".\results\resultsESBMC\ESBMClog.txt"

    # Create directory tree if it does not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)



    max_nb_vertices = 6
    in_dimension_of_hidden_layers=5
    with open(path, "a") as f:
        f.write(f"# Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# test with dimension {in_dimension_of_hidden_layers}, nb of layers = {in_n_layers},activation function-{in_activation}\n")
        base = in_filename.replace(".c", "")  
        for N in range(1, max_nb_vertices+1):
            filename_N = f"{base}_Nbound{N}_testGNN.c"
            start = time.time()
            print(in_configurations)
            T = ESBMCVerificationTask(Nbound = N,type="float",filename=filename_N,activation=in_activation,confifuration_matrices=in_configurations)
            for i in range(in_dimension_of_hidden_layers):
                x = T.add_input_feature()
                for v in range(N):
                    print(f"Adding precondition for vertex {v} and feature {i}")
                    T.add_precondition(f"{x}[{v}] == 0 || {x}[{v}] == 1")
                               
            for i in range(in_n_layers):
                Mvertex = state[f'convs.{i}.V.linear.weight'].t().tolist()
                vertex_bias = state[f'convs.{i}.V.linear.bias'].unsqueeze(0).tolist()
                print('Mvertex',np.shape(Mvertex),np.shape(vertex_bias))
                
                Magg = state[f'convs.{i}.A.linear.weight'].t().tolist()
                agg_bias = state[f'convs.{i}.A.linear.bias'].unsqueeze(0).tolist()
                print('Magg',np.shape(Magg))
                Maggglobal = state[f'convs.{i}.R.linear.weight'].t().tolist()
                aggglobal_bias = state[f'convs.{i}.R.linear.bias'].unsqueeze(0).tolist()
                print('Maggglobal',np.shape(Maggglobal))
                biais = [[vertex_bias[0][j]+agg_bias[0][j]+aggglobal_bias[0][j] for j in range(len(vertex_bias[0]))]]
                print('biais',np.shape(biais))
                
                #Batch normalization parameters a*feature_afterAF+c
                bn_a, bn_c = apply_bn_inference(state, f'batch_norms.{i}')
                print('bn_a',np.shape(bn_a))
                print('bn_c',np.shape(bn_c))
                T.add_layer(Mvertex, Magg, Maggglobal, biais,bn_a, bn_c)
            
            #Linear Prediction parameters> output = feature_afterBN*W+bias
            W = state['linear_prediction.weight'].t().tolist()#https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
            bias = state['linear_prediction.bias'].unsqueeze(0).tolist()
            print('W',np.shape(W))
            print('bias',np.shape(bias))

            T.add_linear_prediction_layer(W, bias)
            print(f"Adding postcondition for N={N}")
            T.add_postcondition(f"{T.get_last_feature()}[0] >= 0")
            
            print("Adding check")
            T.check()
            end = time.time()
            
            f.write(f"Finished N={N} in {end - start:.4f}s\n\n") 
        f.write("\n")
        f.write("\n")






ACRGNN_configurations = ['Cx+Ay+Rz+b', 'xC+yA+zR+b']
configurations = ACRGNN_configurations[1]  #choose either 'Cx+Ay+Rz+b' or 'xC+yA+zR+b'
folder = Path(f"results/resultsESBMC")
folder.mkdir(parents=True, exist_ok=True)

smt_path = folder / f"main.c"
trained_model_fp32(str(smt_path),configurations)



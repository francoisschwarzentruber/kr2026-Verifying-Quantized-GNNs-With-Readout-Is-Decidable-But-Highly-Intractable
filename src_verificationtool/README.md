# ACR-GNNVerification tool
Our goal is to translate an Aggregate Combine Graph Neural Network with the global Readout (ACR-GNN) to the code that we can pass trought Z3 solver

## Graph Neeural Network
In this code we are using the reference architecture of the paper of the Barcelò et. al.[1] with the implementation[2].

## Logic
For the Logic that we can use to verify chosen models is the modal $q\mathcal{L}$ Logic that was described in the article.

## Verification tool(s)
For these experiments we are using two verification tools: ESBMC[3]. So in this case we have a comparison between the SMT solvers.

## Flow from the ESBMC
Here we have a flow of the programms that e are using to have:

Python program -[generate]-> C program -[parse]-> ESBMC solver -[obtain]-> Result: SAT or UNSAT.

>[!Note]
> This repository provides a Python-based tool (main.py) that automatically generates C programs encoding small Aggregate- Combine Graph Neural Network with global readout (ACR-GNN)-like layers and verifies user-defined postconditions using the ESBMC bounded model checker and its underlying SMT solvers.
>The tool supports:
> - Arbitrary graph size (parameter Nbound)
> - Multiple layers
> - Multiple feature vectors
> - Several activation functions: ReLU, trReLU, and ReLU{p}
> - Automatic SMT generation, logging, model extraction.

#### Phase0. Preprocessing.  

1. Create a class VerificationTask.
2. Variable `index_of_feature` set default to 0. We will use it after for indexation of variables.
<details>
  <summary>Sploiler</summary>
 With these index we can apply to postcondition to the feature of the last layer after applying the activation function. These index will be used to call the feature and be sure that all features have unique id.
</details>

#### Phase1. Creating functions.

**Create a function `__init__`.**
As input gets:
- `Nbound`- default value set to `2`.
- `type` - default value set to `charsaturation`.
- `filename` - default value set to `main.c`.
- `activation` -default value set to `ReLU`.
- `combination scheme` -default value set to `Cx+Ay+Rz`. Note: future work

Here we are working with the instancees.

- `Nbound` variable that takes into accound dimention of the unknown graph.
- `filename` variable stores the name of the output file that will be used for the ESBMC.
- `start_of_program` flag for writting the output file.
- `features` stores all names of features
- `activation` store the name of the selected activation function
- `type` - type of the saturation that wil be used. Possible options: `charsaturation` and `float`
- `_headerCprogram()` create a header of the C programm.

**Create function `_headerCprogram`.**

Open the file, write the C program header, and update the flag to indicate that the file exists and should not be overwritten.

**Create function `_addLineInMain`.**

Writes a given string to the specified output file. The resulting file is later consumed by the ESBMC SMT solver.

**Name feature; function `_get_new_featurename`.**
    
Here we are creating the unique name and storing the features that we are using.

This is useful to if we want to know in the end the final feature. (we do!)

**Name feature; function `add_input_feature`.**

Here we have a function that we call from the structure of the ACR-GNN. Basically, declare the feature from the input and later never.

```C
feature(x1);
unknownFeature(x1);
```
**Name feature; function `add_feature`.**
Declare the feature inside the code.

```C
feature(x4);
```

>[!Important] 
> Here is the difference between these two functions. For the nput feature we need to add the additional line that this feature is unknown!

**Preconditions `add_precondition`**

This method inserts a precondition into the encoding of the ACR-GNN verification problem.
A precondition is a logical constraint on existing feature variables (e.g., `x1[0] == 1, x2[3] == 0 || x2[3] == 1`).

**Layer of the ACR-GNN `add_layer`**

As input, this function takes 3 matrices and vector.
- $C$, $A$, $R$ - three matrices that corresponds to $C$ - combination, $A$ - aggregation and $R$ - the readout (global aggregation).
- $b$ - bias vector

> [!WARNING]  
> We use the scheme `C x + A y + R + b`. You must ensure that all
> matrices and vectors have compatible dimensions so that the matrix
> multiplications are well-defined. The same applies to the bias `b`.
>
> By design, each feature vector (e.g., `x1`, `x2`, …) has dimension  
> `Nbound × 1` (a column vector), corresponding to the variables  
> `x1[0] … x1[Nbound-1]` encoded as `x10 … x1(Nbound-1)`.
>
> Therefore, any matrix that multiplies a feature vector from the left
> must have shape `k × Nbound` (for some `k`). In particular, if you
> want a scalar result, you should use a `1 × Nbound` matrix.
> If these dimensions do not match, the generated SMT encoding will no
> longer correspond to a valid matrix–vector product.
To check the validy of the input we are using the finction `checking_input_matrices`

We have the scheme:
- store the input dimension that we get from the C matrix in variable `input_dimension`.
-store the output dimension in variable `input_dimension`.
- Look on the featurs that we are having right now.
- `previousFeatures = self.features[-input_dimension:]` take last Nbound number of features
- declare the local and global aggregation variables. Mark them into output file.
- block of the matrix multiplication

> [!Important] 
> Encodes, for each position $i = 0 \cdots Nbound-1$ and each output o:
> 
>  $$u_o(i) = sum_j C[o][j] * x_j(i)+ sum_j A[o][j] * y_j(i)+ sum_j R[o][j] * z_j(i)+ b[o][0]$$
> 
>  where:
> - previousFeatures[j]      = base name of x_j (e.g., "x1")
> - aggPreviousFeatures[j]   = base name of y_j
> - aggGPreviousFeatures[j]  = base name of z_j
> - outputFeatures[o]        = base name of u_o (e.g., "x5", "x6")

- apply activation function $\alpha(u_o(i))$. Possible options: 'ReLU', 'trReLU' and 'ReLU{param}'(e.g. ReLU6, ReLU2, etc.).

> [!Important] 
> The code is strictly bound to the register naming, so activation functions must be written according to the supported options. For ReLU{param}, a number can be specified after ReLU; this parameter determines the clipping threshold for the positive part of the function.


**function `add_linear_prediction_layer`**
Applying the linear prediction layer after the last layer. Correspond for the classification. This function is optional if we need to verify property before.

**function `get_last_feature`.**  

As was mention in the name of the function return the last feature 


**Postconditions; function`add_postcondition`.**

The same flow as Preconditon block, but we also check if the feature is exists. To prevent possivle errors to double the information we added the checker.

**function `_endCprogram`.**  

Added to the `.c` file the ending of the C code.

**Check the model; function `check`**

Run ESBMC on the generated C file and return the verification result.

Returns:
- status: 'sat'(Counterexample found), 'unsat'(Verification passed)
- values: generate file `.cex.txt` with the steps if sat, empty dict otherwise.

## Reference

[1] Pablo Barceló, Egor V. Kostylev, Mikaël Monet, Jorge Pérez, Juan L. Reutter, and Juan Pablo Silva.  
**The Logical Expressiveness of Graph Neural Networks**, 8th International Conference on Learning Representations (ICLR), 2020.  
Available at: [https://openreview.net/forum?id=r1lZ7AEKvB](https://openreview.net/forum?id=r1lZ7AEKvB)

[2] Pablo Barceló, Egor V. Kostylev, Mikaël Monet, Jorge Pérez, Juan L. Reutter, and Juan Pablo Silva.  
**GNN-logic**, GitHub repository, 2021.  
Available at: [https://github.com/juanpablos/GNN-logic](https://github.com/juanpablos/GNN-logic)

[3] Menezes, R., Aldughaim, M., Farias, B., Li, X., Manino, E., Shmarov, F., Song, K., Brauße, F., Gadelha, M. R., Tihanyi, N., Korovin, K., & Cordeiro, L. C. **ESBMC 7.4: Harnessing the Power of Intervals**, TACAS, LNCS 14572, pp. 376–380. Springer,2024.  
Available at: [https://doi.org/10.1007/978-3-031-57256-2_24](https://doi.org/10.1007/978-3-031-57256-2_24)
    

## License

This project is licensed under the [MIT License](LICENSE).

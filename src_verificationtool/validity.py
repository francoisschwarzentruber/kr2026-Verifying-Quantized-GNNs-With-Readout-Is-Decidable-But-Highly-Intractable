import numpy as np

def checking_input_matrices(input_dimension, C, A, R, b, configurations):
    matrices = [C, A, R]
    # all matrices have the same number of columns
    check_cols = []
    check_rows = []
    for M in matrices:
        Nrows, Ncolumns = np.shape(M)
        check_cols.append(Ncolumns)
        check_rows.append(Nrows)

    if configurations == 'Cx+Ay+Rz+b':
        # all columns must equal dimension k times input_dimension (number of input features) and correspond to the dimension of the feature vector. 
        # Feature vector has dimension Nbound times 1. So number of columns of the input matrices must be equal to Nbound
        print(set(check_cols))
        if len(set(check_cols)) != 1 or check_cols[0] != input_dimension:
            raise ValueError("Invalid input! Matrices must have shape k × input_dimension (number of input features).")

        # all rows must be equal (same k for C, A, R)
        if len(set(check_rows)) != 1:
            raise ValueError("Not equal shapes of rows of the matrices (C, A, R).")

        k = check_rows[0]

        #bias shape: must be k × 1
        if np.shape(b)[0] != k:
            raise ValueError("Not equal shapes of rows between matrices and bias.")
        if np.shape(b)[1] != 1:
            raise ValueError("Bias has a wrong column shape; expected (k, 1).")
    elif configurations == 'xC+yA+zR+b':
        # all rows must equal input_dimension
        if len(set(check_rows)) != 1 or check_rows[0] != input_dimension:
            raise ValueError("Invalid input! Matrices must have shape input_dimension × k.")

        # all rows must be equal (same k for C, A, R)
        if len(set(check_cols)) != 1:
            raise ValueError("Not equal shapes of columns of the matrices (C, A, R).")
        
        # rows can be different for C, A, R
        k = check_cols[0]
        # bias shape: must be 1 × k
        if np.shape(b)[0] !=1:
            raise ValueError("Not equal shapes of rows between matrices and bias.")
        if np.shape(b)[1] != k:
            raise ValueError("Bias has a wrong column shape; expected (1, k).")
    else:
        raise ValueError("Unsupported configuration string.")
    return 'fine'

'''
#test the code
C = [[1, 2]]
A = [[0, 0]]
R = [[0, 0]]
b = [[0]]
print("Testing checking_input_matrices function...")
print("Input matrices:")
ACRGNN_configurations = ['Cx+Ay+Rz+b', 'xC+yA+zR+b']
configurations = ACRGNN_configurations[1]
input_dimension = len(C[0])
output_dimension = len(C)
print("Input dimension:", input_dimension)
print("Output dimension:", output_dimension)    
# validity
print(checking_input_matrices(input_dimension, C, A, R, b, configurations))
'''
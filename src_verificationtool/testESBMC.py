import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
from pathlib import Path


from ESBMCVerificationTask import ESBMCVerificationTask



def simpleACRGNN(in_filename,in_activation):
    nbound=2
    in_filename= in_filename.replace('.c',f'Nbound{nbound}_simpleACRGNN_{in_activation}.c')
    v = ESBMCVerificationTask(nbound,filename=in_filename,activation=in_activation)
    for i in range(nbound):
        v.add_input_feature()
    #add preconditions
    v.add_precondition("x1[0] == 1")
    v.add_precondition("x2[0] == 2")
    #add layer with the C, A, R ,b matrices  
    v.add_layer([[1, 2]],
                [[0, 0]],
                [[0, 0]],
                [[0]])
    
    #add postconditions
    v.add_postcondition(f"{v.get_last_feature()}[0] == 5")
    status, values = v.check()
    print("STATUS:", status)

    if status == "sat":
        print("INTERMEDIATE VALUES:")
        for name, val in sorted(values.items()):
            print(f"  {name} = {val}")
    else:
        print("No model available (unsat/unknown).")


def simpleACRGNN_bias(in_filename,in_activation):
    nbound=2
    in_filename= in_filename.replace('.c',f'Nbound{nbound}_simpleACRGNN_bias_{in_activation}.c')
    v = ESBMCVerificationTask(Nbound = nbound,filename=in_filename,activation=in_activation)
    for i in range(nbound):
        v.add_input_feature()
    #add preconditions
    v.add_precondition("x1[0] == 1")
    v.add_precondition("x2[0] == 2")
    #add layer with the C, A, R ,b matrices
    v.add_layer([[1, 2]],
                [[0, 0]],
                [[0, 0]],
                [[5]])
    #add postconditions
    v.add_postcondition(f"{v.get_last_feature()}[0] == 0")
    v.check()

def justRunATest(in_filename,in_activation):
    """
    small example of how to use the tool
    """
    nbound=3
    in_filename= in_filename.replace('.c',f'Nbound{nbound}_justRunATest_{in_activation}.c')
    T = ESBMCVerificationTask(Nbound = 3,filename=in_filename,activation=in_activation)
    for i in range(3):
        T.add_input_feature()
    T.add_precondition("x1[0] == 0")
    T.add_precondition("x1[1] == 0")
    T.add_precondition("x1[2] == 0")

    T.add_precondition("x2[0] == 0")
    T.add_precondition("x2[1] == 0")
    T.add_precondition("x2[2] == 0")
    
    T.add_precondition("x3[0] == 0 || x3[0] == 1")
    T.add_precondition("x3[1] == 0")
    T.add_precondition("x3[2] == 0")

    T.add_layer([[2, 3, 1], [1, 0, -7]],
                [[2, 3, 1], [1, 0, -7]],
                [[2, 3, 1], [1, 0, -7]],
                [[1], [8]])
    T.add_postcondition(f"{T.get_last_feature()}[0] >= 0")
    T.check()

def testGNN(in_filename,in_activation):
    dimension =2 
    nb_layers = 2
    max_nb_vertices = 6
    with open(f"results/resultsESBMC/ESBMClog.txt", "a") as f:
        f.write(f"# Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# test with dimension {dimension}, nb of layers = {nb_layers},activation function-{in_activation}\n")
        base = in_filename.replace(".c", "")  
        for N in range(1, max_nb_vertices+1):
            filename_N = f"{base}_Nbound{N}_testGNN.c"
            start = time.time()
            T = ESBMCVerificationTask(Nbound = N,filename=filename_N,activation=in_activation)
            for i in range(dimension):
                x = T.add_input_feature()
                for v in range(N):
                    T.add_precondition(f"{x}[{v}] == 0 || {x}[{v}] == 1")
                               
            for i in range(nb_layers):
                Mvertex = [[random.randint(1, 10) for _ in range(dimension)] for _ in range(dimension)]
                print('Mvertex',np.shape(Mvertex))
                Magg = [[random.randint(1, 10) for _ in range(dimension)] for _ in range(dimension)]
                print('Magg',np.shape(Magg))
                Maggglobal = [[random.randint(1, 10) for _ in range(dimension)] for _ in range(dimension)]
                print('Maggglobal',np.shape(Maggglobal))
                biais = [[random.randint(1, 10)] for _ in range(dimension)]
                print('biais',np.shape(biais))
                T.add_layer(Mvertex, Magg, Maggglobal, biais)

            T.add_postcondition(f"{T.get_last_feature()}[0] >= 0")

            T.check()
            end = time.time()
            f.write(f"Finished N={N} in {end - start:.4f}s\n\n") 
        f.write("\n")
        f.write("\n")





ACRGNN_configurations = ['Cx+Ay+Rz+b', 'xC+yA+zR+b']
configurations = ACRGNN_configurations[0]  #choose either 'Cx+Ay+Rz+b' or 'xC+yA+zR+b'
activations = ['ReLU','ReLU6','trReLU']
folder = Path(f"results/resultsESBMC")
folder.mkdir(parents=True, exist_ok=True)
'''
for act in activations:
    folder_act = Path(f"{folder}/results_{act}")
    folder_act.mkdir(parents=True, exist_ok=True)
    smt_path = folder_act / f"main_{act}.c"
    testGNN(str(smt_path), act)
'''
smt_path = folder / f"main.c"
#simpleACRGNN(str(smt_path),'ReLU')
simpleACRGNN_bias(str(smt_path),'ReLU')
#justRunATest(str(smt_path),8,'ReLU')


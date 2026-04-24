# Verification tool for quantized GNNs

## Problem tackled by the tool

The verification problem is as follows. 

The input of the problem is:
- an integer N which is a bound on the number of vertices in the graph counterexample
- a precondition, e.g. x1[0] == 1 && (x1[1] == 1 || x2[1] == 1) which means feature x1 at node 0 is 1 and either feature x1 or x2 at node 1 is at 1.
- a GNN specified by the layers. Each layer is specified by a matrix on the local features, a matrix on the aggregation of successors, and a matrix on the global readout.
- a postcondition e.g. x10[0] >= 0, meaning that feature x10 computed by the GNN is positive

The output is yes if for all graphs of size at most N satisfying the precondition, the output of the GNN satisfies the postcondition), and no otherwise.
Furthermore, the user can extract a countermodel when the answer is no from the output of the model checker.


## Setup

- Get the executable of ESBMC (Efficient SMT-based Context-Bounded Model Checker) available in release v7.9 here: https://github.com/esbmc/esbmc 
- Put the executable `esbmc` in the current folder
- Modify `main.py` to specify your verification task 


## Type checking

You can type-check the Python program as follows:
- `mypy main.py`

## Usage

When running `python main.py`, the program creates the C program `main.c` that simulates the computation of the GNN. Preconditions are added via the directive `__VERIFIER_assume`, e.g. `__VERIFIER_assume(x1[0] == 1 && (x1[1] == 1 || x2[1] == 1));`. The postcondition is added via `assert`, e.g. `assert(x10[0] >= 0);`.

Here is a example of C program produced by the tool:


```c
#define N 3
#include "typecharsaturation.h"
#include "quantlogic.h"
int main()
{
    testNumber();
    unknownGraph();
    feature(x1);
    unknownFeature(x1);
    feature(x2);
    unknownFeature(x2);
    feature(x3);
    unknownFeature(x3);
    __VERIFIER_assume(x1[0] == 0);
    __VERIFIER_assume(x1[1] == 0);
    __VERIFIER_assume(x1[2] == 0);
    __VERIFIER_assume(x2[0] == 0);
    __VERIFIER_assume(x2[1] == 0);
    __VERIFIER_assume(x2[2] == 0);
    __VERIFIER_assume(x3[0] == 0 || x3[0] == 1);
    __VERIFIER_assume(x3[1] == 0);
    __VERIFIER_assume(x3[2] == 0);
    feature(x4);
    feature(x5);
    feature(x6);
    feature(x7);
    feature(x8);
    feature(x9);
    agg(x4, x1);
    aggG(x7, x1);
    agg(x5, x2);
    aggG(x8, x2);
    agg(x6, x3);
    aggG(x9, x3);
    feature(x10);
    mul(x10, 2, x1);
    mul(x10, 3, x2);
    mul(x10, 1, x3);
    mul(x10, 2, x4);
    mul(x10, 3, x5);
    mul(x10, 1, x6);
    mul(x10, 2, x7);
    mul(x10, 3, x8);
    mul(x10, 1, x9);
    addCte(x10, -1);
    reLU(x10, x10);
    feature(x11);
    mul(x11, 1, x1);
    mul(x11, 0, x2);
    mul(x11, -7, x3);
    mul(x11, 1, x4);
    mul(x11, 0, x5);
    mul(x11, -7, x6);
    mul(x11, 1, x7);
    mul(x11, 0, x8);
    mul(x11, -7, x9);
    addCte(x11, -8);
    reLU(x11, x11);
    assert(x10[0] >= 0);
    return 0;
}
```

This C program is then verified with `esbmc` via `esbmc main.c`.



## License

This project is licensed under the [MIT License](LICENSE).
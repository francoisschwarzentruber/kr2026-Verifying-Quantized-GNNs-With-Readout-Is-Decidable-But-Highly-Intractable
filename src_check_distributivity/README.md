# Verification of the distributivity law

This folder contains a C program `main.c` in order to check that (1 + 1 + .... + 1) * f = (f + f + ... + f) for some rings.


## Setup

- Get the executable of ESBMC (Efficient SMT-based Context-Bounded Model Checker) available in release v7.9 here: https://github.com/esbmc/esbmc 
- Put the executable `esbmc` in the current folder

## Usage

This C program is verified with `esbmc` via `esbmc main.c`.



## License

This project is licensed under the [MIT License](LICENSE).
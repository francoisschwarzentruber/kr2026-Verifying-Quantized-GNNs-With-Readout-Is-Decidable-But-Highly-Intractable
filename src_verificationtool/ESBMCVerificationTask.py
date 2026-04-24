""" This program creates verification tasks for GNNs. It checks

Python program -[generate]-> C program -[parse]-> ESBMC solver -[obtain]-> Result: SAT or UNSAT.

Returns:
    _type_: _description_
"""

import subprocess
from pathlib import Path
import sys
from pathlib import Path as PathlibPath


import re

# Add parent directory to path to import validity
sys.path.insert(0, str(PathlibPath(__file__).parent.parent))

from validity import checking_input_matrices

Number = int | float

def parse_esbmc_counterexample(text: str) -> dict[str, int]:
    values: dict[str, int] = {}

    # matches: x7[0] = -5   OR   x7[0]=-5
    pat = re.compile(r"\b(x\d+\[\d+\])\s*=\s*(-?\d+)\b")

    for m in pat.finditer(text):
        values[m.group(1)] = int(m.group(2))

    # optional: adjacency booleans like e0_1 = TRUE / FALSE
    pat_e = re.compile(r"\b(e\d+_\d+)\s*=\s*(TRUE|FALSE)\b", re.IGNORECASE)
    for m in pat_e.finditer(text):
        values[m.group(1)] = 1 if m.group(2).upper() == "TRUE" else 0

    return values

class ESBMCVerificationTask:
    index_of_feature = 0
    
    
    def __init__(self, Nbound = 2, type="charsaturation",filename="main.c",activation='ReLU',confifuration_matrices='Cx+Ay+Rz+b'):
        """ Initialize a new verification task

        Args:
            Nbound (int, optional): bound on the number of vertices in the example/counterexample we are search for. Defaults to 3.
            type (str): a string that is the name of a C type.
                        Can be either "float" or "charsaturation".
        """
        self.Nbound = Nbound
        self.filename = filename
        self.start_of_program = "w"
        self.features = []
        self.activation =activation
        self.type = type
        self._headerCprogram()
        self.configuration_matrices = confifuration_matrices
        
    def _headerCprogram(self) -> None:
        """ add the header of the C program
        """
        with open(self.filename, self.start_of_program) as c_file:
            c_file.write(f"#define Nbound {self.Nbound}\n")
            c_file.write('unsigned int N = Nbound; //number of vertices\n')
            c_file.write(f'#include "type{self.type}.h"\n')
            c_file.write('#include "quantlogic.h"\n\n')
            c_file.write("int main()\n")
            c_file.write("  {\n")
            c_file.write("  testNumber();\n")
            c_file.write("  for(int N1 = 1; N1 <= Nbound; N1++)\n") #loop over possible size of graphs
            c_file.write("  {\n") # { of the for loop  
            c_file.write("    N = N1;\n") #assign the number of vertices N (global variable)
            c_file.write("    unknownGraph();\n")
        self.start_of_program ="a"

    def _addLineInMain(self, line: str) -> None:
        """ add a line in the main function in the C program

        Args:
            line (str): line to be added to the main function in the generated C program
        """
        with open(self.filename, self.start_of_program) as c_file:
            c_file.write("    " + f"{line}\n")
        
    
    def _get_new_featurename(self) -> str:
        """ Create a new feature name (e.g. "x3", "x4", etc.)

        Returns:
            str: the name of the feature, e.g. "x3", "x4", etc.
        """
        self.index_of_feature = self.index_of_feature + 1
        featureName = f"x{self.index_of_feature}"
        self.features.append(featureName)
        return featureName

    def add_input_feature(self) -> str:
        """ Add a new input feature. An input feature is undefined. We are search for a value of it in a counterexample.

        Returns:
            str: the name of the feature that was added
        """
        x = self._get_new_featurename()
        self._addLineInMain(f"feature({x});")
        self._addLineInMain(f"unknownFeature({x});")
        return x
    
    def _add_feature(self) -> str:
        """ Add an intermediate feature (set to 0 initially)

        Returns:
           str: name of the added feature
        """
        x = self._get_new_featurename()
        self._addLineInMain(f"feature({x});")
        return x
        
    def add_precondition(self, precondition: str) -> None:
        """ add a precondition

        Args:
            precondition (str): a string representing a precondition in C. 
            For example, "x1[0] <= 5 && x2[1] >= 6"
        """
        self._addLineInMain(f"__ESBMC_assume({precondition});")

    def add_layer(self,C: list[list[Number]],
                  A: list[list[Number]], 
                  R: list[list[Number]],
                  b: list[list[Number]],
                  a: list[Number]| None = None,
                  c: list[Number]| None = None) -> None:
        
        if self.configuration_matrices == 'Cx+Ay+Rz+b':
            input_dimension = len(C[0])
            output_dimension = len(C)
        elif self.configuration_matrices == 'xC+yA+zR+b':
            input_dimension = len(C)
            output_dimension = len(C[0])
        else:
            raise ValueError("Configuration wrong or unsupported. Supported: 'Cx+Ay+Rz+b' or  'xC+yA+zR+b'")
        
        # validity
        if checking_input_matrices(input_dimension, C, A, R, b,self.configuration_matrices) != 'fine':
            raise ValueError("Something wrong. Check your input dimensions.")

        previousFeatures = self.features[-input_dimension:]
        if len(previousFeatures) != input_dimension:
            raise ValueError("Not enough features allocated for input_dimension.")
        
        aggPreviousFeatures = [self._add_feature() for j in range(input_dimension)]
        aggGPreviousFeatures = [self._add_feature() for j in range(input_dimension)]
        
        for j in range(input_dimension):
            self._addLineInMain(f"agg({aggPreviousFeatures[j]}, {previousFeatures[j]});")
            self._addLineInMain(f"aggG({aggGPreviousFeatures[j]}, {previousFeatures[j]});")

        for i in range(output_dimension):
            outputFeatures = self._add_feature()
            if self.configuration_matrices == 'Cx+Ay+Rz+b':
                for j in range(input_dimension):
                    self._addLineInMain(f"mul({outputFeatures}, {C[i][j]}, {previousFeatures[j]});")
                    
                for j in range(input_dimension):
                    self._addLineInMain(f"mul({outputFeatures}, {A[i][j]}, {aggPreviousFeatures[j]});")  
                    
                for j in range(input_dimension):
                    self._addLineInMain(f"mul({outputFeatures}, {R[i][j]}, {aggGPreviousFeatures[j]});")
                    
                self._addLineInMain(f"addCte({outputFeatures}, {b[i][0]});")
            elif self.configuration_matrices == 'xC+yA+zR+b':
                for j in range(input_dimension):
                    self._addLineInMain(f"mul({outputFeatures}, {C[j][i]}, {previousFeatures[j]});")
                    
                for j in range(input_dimension):
                    self._addLineInMain(f"mul({outputFeatures}, {A[j][i]}, {aggPreviousFeatures[j]});")  
                    
                for j in range(input_dimension):
                    self._addLineInMain(f"mul({outputFeatures}, {R[j][i]}, {aggGPreviousFeatures[j]});")
                    
                self._addLineInMain(f"addCte({outputFeatures}, {b[0][i]});")
            else:
                raise ValueError("Configuration wrong or unsupported. Supported: 'Cx+Ay+Rz+b' or  'xC+yA+zR+b'")
            
            if self.activation == "ReLU":
                self._addLineInMain(f"{self.activation}({outputFeatures}, {outputFeatures});")
            elif self.activation.startswith("ReLU"):
                param_str = self.activation[4:]  # after 'ReLU'
                if not param_str.isdigit():
                    raise ValueError(f"Unsupported ReLU variant: {self.activation}")
                if param_str.isdigit():
                    param = int(param_str)
                    self._addLineInMain(f"ReLUp({outputFeatures}, {outputFeatures}, {param});")
            elif self.activation == "trReLU":
                self._addLineInMain(f"trReLU({outputFeatures}, {outputFeatures});")
            else:
                raise ValueError(
                    "Activation function wrong or unsupported. "
                    "Supported: ReLU, ReLU{p} (e.g. ReLU6, ReLU2, etc.), trReLU"
                )
            if a is not None and c is not None:
                #as in the gitrepository BN was applied after Activation function according to the architecture choice.
                self._addLineInMain(f"mul({outputFeatures}, {a[i]}, {outputFeatures});")
                self._addLineInMain(f"addCte({outputFeatures}, {c[i]});")
            
    def add_linear_prediction_layer(self, W: list[list[Number]], bias: list[list[Number]]):
        if self.configuration_matrices == 'Cx+Ay+Rz+b':
            input_dimension = len(W[0])
            output_dimension = len(W)
        elif self.configuration_matrices == 'xC+yA+zR+b':
            input_dimension = len(W)
            output_dimension = len(W[0])
        else:
            raise ValueError("Configuration wrong or unsupported. Supported: 'Cx+Ay+Rz+b' or  'xC+yA+zR+b'")
        
        
        previousFeatures = self.features[-input_dimension:]
        if len(previousFeatures) != input_dimension:
            raise ValueError("Not enough features allocated for input_dimension.")
        if self.configuration_matrices == 'Cx+Ay+Rz+b':
            for i in range(output_dimension):
                outputFeature = self._add_feature()
                for j in range(input_dimension):
                    self._addLineInMain(f"mul({outputFeature}, {W[i][j]}, {previousFeatures[j]});")
                self._addLineInMain(f"addCte({outputFeature}, {bias[i][0]});")
        elif self.configuration_matrices == 'xC+yA+zR+b':
            for i in range(output_dimension):
                outputFeature = self._add_feature()
                for j in range(input_dimension):
                    self._addLineInMain(f"mul({outputFeature}, {W[j][i]}, {previousFeatures[j]});")
                self._addLineInMain(f"addCte({outputFeature}, {bias[0][i]});")
        else:
            raise ValueError("Configuration not supported.")
       

    def get_last_feature(self) -> str:
        return self.features[-1]
    
    def add_postcondition(self, postcondition: str) -> None:
          self._addLineInMain(f"assert({postcondition});")

    def _endCprogram(self) -> None:
        """ write the C program corresponding to the verification task

        Returns:
            _type_: _description_
        """

        """
        self.Cprogram.append('  }') # end of the for loop
        self.Cprogram.append('  return 0;')
        self.Cprogram.append('}') # end of the C main function
        """

        # Check if the ending is already written
        try:
            with open(self.filename, 'r') as f:
                content = f.read()
                if content.endswith('  }\n  return 0;\n}\n'):
                    return  # Already ended
        except FileNotFoundError:
            pass

        with open(self.filename, self.start_of_program) as c_file:
            c_file.write('  }\n')
            c_file.write('  return 0;\n')
            c_file.write('}\n')

    def check(self):
        """
        Run ESBMC on the generated C file and return the verification result.
        
        Returns:
            status: 'sat' if a counterexample is found, 'unsat' if verification passes, 'unknown' otherwise
            values: dict with variable values if sat, empty dict otherwise
        """
        self._endCprogram()
        
        print(self.filename)
        c_file  = self.filename
        INC_DIR = Path(__file__).resolve().parent
        cex_file = str(Path(c_file).with_suffix(".cex.txt"))
        proc = subprocess.run(
            [
                'esbmc',
                "--no-library",
                "-I", str(INC_DIR),
                "--no-bounds-check",
                "--no-pointer-check",
                "--no-div-by-zero-check",
                "--z3",
                "--cex-output",cex_file,
                str(Path(c_file).resolve()),
            ],
            capture_output=True,
            text=True
        )
        
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")

        # status detection (robust to return codes)
        if "VERIFICATION SUCCESSFUL" in out:
            print("ESBMC output indicates verification successful.")
            return "unsat", {}
        if "VERIFICATION FAILED" in out:
            print("ESBMC output indicates verification failed. Check '.cex.txt' file to know more.")
            # parse counterexample either from file (preferred) or from output text
            text = Path(cex_file).read_text(errors="ignore") if Path(cex_file).exists() else out
            values = parse_esbmc_counterexample(text)  # implement below
            return "sat", values

        return "unknown", {}
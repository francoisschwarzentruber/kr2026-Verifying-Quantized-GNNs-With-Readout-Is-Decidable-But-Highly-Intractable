import subprocess

with open(r"Supplement_materials\Code\src\requirements.txt", "w") as f:
    subprocess.run(["pip", "freeze"], stdout=f)
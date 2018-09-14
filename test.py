import sys
import os
from pathlib import Path

def findRootDir(path, dirname):
    path = Path(path)
    if os.path.basename(path.parent) == dirname:
        return path.parent
    else:
        return findRootDir(path.parent, dirname)

root_path = findRootDir(os.path.realpath(__file__), "AI_playground")
os.chdir(root_path)
sys.path.insert(0, root_path)

try:
    with open("./notebooks/global_setup.py") as setupfile:
        exec(setupfile.read())
except FileNotFoundError:
    print('Setup already completed')

a = 1
b = []
a + b

# %%
import os
import sys
import pathlib
import torch

BASE = pathlib.Path(__file__).parents[1]
LIB_HOME = BASE / "lib"
DATA_HOME = BASE / "data"

SMALL_NUMBER = 1e-6
DTYPE = torch.get_default_dtype()

# %%

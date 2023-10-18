import os
import sys
import pickle as pk
import random
import math
import pandas as pd
import numpy as np

from GSMM import *
from obj_fxn_maxwtclique import *



def pk_save(obj, fname):
    with open(fname, 'wb') as handle:
        pk.dump(obj, handle, protocol=pk.HIGHEST_PROTOCOL)

def pk_load(fname):
    with open(fname, 'rb') as handle:
        return pk.load(handle)
    
def check_device():
    import torch
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available.")









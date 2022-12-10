import numpy as np 
import torch
import torch.nn.functional as nnf

x = torch.rand(5, 1, 44, 44)
out = x.resize(5, 1, 45, 44)
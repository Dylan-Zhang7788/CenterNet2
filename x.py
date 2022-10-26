import torch

a=torch.tensor([0,0,0,1,1,3,0])
b=torch.nonzero(a)
print(b)
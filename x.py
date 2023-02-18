import torch

sizes_of_interest=[[0, 64], [48, 192], [128, 1000000]]
x=torch.tensor([0, 64]).float().view(
1, 2)
print(x)
#.expand(num_loc_list[l], 2) for l in range(L)]) # [M,2]
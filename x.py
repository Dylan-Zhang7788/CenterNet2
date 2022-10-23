import torch 

feature=([4,2],[8,4],[16,8])
strides=(8, 16,32)

def compute_grids(strides, features):
    grids = []
    for level, feature in enumerate(features):
        print(feature)
        h, w = feature[0],feature[1]
        # 默认strides=(8, 16, 32, 64, 128)
        shifts_x = torch.arange(
            0, w * strides[level], 
            step=strides[level],
            dtype=torch.float32)
        shifts_y = torch.arange(
            0, h * strides[level], 
            step=strides[level],
            dtype=torch.float32)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        print(shifts_x)
        print(shifts_y)
        print(shift_x)
        print(shift_y)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        print(shift_x)
        print(shift_y)
        grids_per_level = torch.stack((shift_x, shift_y), dim=1)
        # grids_per_level = torch.stack((shift_x, shift_y), dim=1) + \
        #     strides[level] // 2
        print(grids_per_level)
        grids.append(grids_per_level)
    return grids

print(compute_grids(strides,feature))
# compute_grids(strides,feature)

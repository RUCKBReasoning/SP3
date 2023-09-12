import os
import torch
import torch.nn as nn

path0 = "/data2/hyx/projects/tcsp_v2/cache/checkpoints/bert/[glue]-[mrpc]"
path1 = "/data2/hyx/projects/tcsp_v2/cache/checkpoints/bert/[glue]-[mrpc]-[0.01]"

dict1 = torch.load(os.path.join(path0, "pytorch_model.bin"))
dict2 = torch.load(os.path.join(path1, "pmodel_final.bin"))

num_params1 = 0
for name, params in dict1.items():
    num_params1 += params.view(-1).shape[0]

num_params2 = 0
for name, params in dict2.items():
    num_params2 += params.view(-1).shape[0]

num_params3 = 0
num_params4 = 0
num_params5 = 0
for name, params in dict2.items():
    if 'encoder' not in name:
        print("name={}".format(name))
        num_params3 += params.view(-1).shape[0]
    else:
        if 'residual' in name:
            num_params4 += params.view(-1).shape[0]
        else:
            num_params5 += params.view(-1).shape[0]

print(num_params1)
print(num_params2)
print(num_params3)
print(num_params4)
print(num_params2 / num_params1)
print(num_params3 / num_params1)
print(num_params4 / num_params1)
print(num_params5 / num_params1)

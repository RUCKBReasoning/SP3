import os
import torch
import torch.nn as nn

path = "/data2/hyx/projects/tcsp_v2/cache/checkpoints/bert/[glue]-[mrpc]"

dict1 = torch.load(os.path.join(path, "pytorch_model.bin"))
dict2 = torch.load(os.path.join(path, "pmodel_final.bin"))

num_params = 0
for name, params in dict1.items():
    num_params += params.view(-1).shape[0]
print(num_params)

num_params = 0
for name, params in dict2.items():
    num_params += params.view(-1).shape[0]
print(num_params)

import os
import torch
import torch.nn as nn

path = "/data2/hyx/projects/tcsp_v2/cache/checkpoints/bert/[glue]-[mrpc]-[0.05]"

dict1 = torch.load(os.path.join(path, "pytorch_model.bin"))
dict2 = torch.load(os.path.join(path, "pmodel_final.bin"))

num_params1 = 0
for name, params in dict1.items():
    num_params1 += params.view(-1).shape[0]

num_params2 = 0
for name, params in dict2.items():
    num_params2 += params.view(-1).shape[0]
    print(name, params.shape)
a = num_params1 - 23440896 - 512 * 768
b = num_params2 - 23440896 - 512 * 768
print(num_params1 - 23440896 - 512 * 768)
print(num_params2 - 23440896 - 512 * 768)
print(b / a)

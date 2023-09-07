import torch
import torch.nn as nn
import evaluate
import pickle

import datetime

format_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

print(format_time)

# l = nn.Linear(3, 5)
# print(l.weight.shape)

# class B(nn.Module):
    
#     def __init__(self) -> None:
#         super().__init__()
#         self.comp_module_dict = nn.ModuleDict()
#         self.lb = nn.Linear(1, 1)
    

# class A(nn.Module):

#     def __init__(self) -> None:
#         super().__init__()
#         self.comp_module_dict = nn.ModuleDict()
#         self.l = nn.Linear(20, 20)
#         self.b = B()
#         self.d = nn.ModuleDict({'ll': nn.Linear(3, 3)})


# a = A()
# print(list(a.named_parameters()))
# print(list(a.named_modules()))

# print(list(a.named_children()))

# a.cuda()

# a.l2 = nn.Linear(10, 10)
# print(list(a.named_children()))

# print(a.l.weight.device)
# print(a.l2.weight.device)

# a.cuda()
# print(a.l.weight.device)
# print(a.l2.weight.device)

# a.use_comp = True
# print(list(a.named_children()))

# """
# output:
# [('l', Linear(in_features=10, out_features=10, bias=True))]
# """

from fvcore.nn import FlopCountAnalysis, parameter_count_table
import os
import torch
from modeling.modeling_packed_bert import (
    PackedBertForSequenceClassification as PModel
)

s_output_dir = "/data2/hyx/projects/tcsp_v2/cache/checkpoints/bert/[glue]-[mnli]-[0.06]"
p_model_path = os.path.join(s_output_dir, "pmodel_init.bin")
p_model_config_path = os.path.join(s_output_dir, "pmodel_config.bin")
p_config = torch.load(p_model_config_path)
p_model = PModel(p_config)
p_model.load_state_dict(torch.load(p_model_path, map_location="cpu"))

input_ids = torch.randint(0, 1000, size=(1, 512))

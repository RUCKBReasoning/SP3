import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EvalPrediction,
    TrainerCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    DataCollator,
)
from typing import Dict, List, Any, Tuple, Callable, Union, Optional, Sequence
from loguru import logger
from tqdm import tqdm


from transformers import Trainer as DefaultTrainer
from transformers.trainer import (
    unwrap_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
)

from modeling.compactor import Mask
from modeling.modeling_compact_bert import (
    CompactBertForSequenceClassification as SModel
)


class DistillTrainerCallback(TrainerCallback):
    
    def __init__(self, args, trainer: DefaultTrainer) -> None:
        self.args = args
        self.trainer = trainer
        self.pruning_start_epoch = self.args.pruning_start_epoch
        self.pruning_warmup_epoch = self.args.pruning_warmup_epoch
        self.pruning_steps = 0
        self.pruning_warmup_steps = None
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        train_dataloader = self.trainer.get_train_dataloader()
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        self.pruning_warmup_steps = self.pruning_warmup_epoch * num_update_steps_per_epoch       

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch >= self.pruning_start_epoch:
            self.trainer.reg_switch = True
            self.trainer.set_reg_params_state(True)
        
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch >= self.pruning_start_epoch:
            self.pruning_steps += 1
            self.trainer.reg_warmup_percent = min(1.0, self.pruning_steps / self.pruning_warmup_steps)


class DistillTrainer(DefaultTrainer):
    
    def __init__(self, 
        s_model: Union[PreTrainedModel, nn.Module] = None, 
        t_model: Union[PreTrainedModel, nn.Module] = None, 
        args: TrainingArguments = None, 
        data_collator: Optional[DataCollator] = None, 
        train_dataset: Optional[Dataset] = None, 
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None, 
        tokenizer: Optional[PreTrainedTokenizerBase] = None, 
        model_init: Optional[Callable[[], PreTrainedModel]] = None, 
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None, 
        callbacks: Optional[List[TrainerCallback]] = None, 
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None), preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    ):
        assert callbacks is None
        callbacks = [DistillTrainerCallback(args, self)]
        super().__init__(
            s_model, args, data_collator, train_dataset, eval_dataset, 
            tokenizer, model_init, compute_metrics, callbacks, optimizers, 
            preprocess_logits_for_metrics
        )
        self.t_model = t_model
        device = next(self.model.parameters()).device
        self.t_model.to(device)
        self.t_model.eval()
        
        self.distill_switch = False
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.mse_loss = nn.MSELoss()

        self.reg_switch = False
        self.reg_warmup_percent = 0.
        self.reg_params = []
        self.reg_z_groups = []
        self.init_reg_params()
        self.set_reg_params_state(False)
        
    def init_reg_params(self):
        for name, _ in self.model.named_parameters():
            if name.endswith('reg_lambda_1') or \
                name.endswith('reg_lambda_2') or \
                name.endswith('log_alpha'):
                self.reg_params.append(name)
        model: SModel = self.model
        past_layer_norm = model.bert.embeddings.LayerNorm
        for layer in model.bert.encoder.layer:
            layer_mask0 = past_layer_norm.mask
            layer_mask1 = layer.attention.output.LayerNorm.mask
            layer_mask2 = layer.output.LayerNorm.mask
            qk_mask = layer.attention.self.query.mask
            vo_mask = layer.attention.self.value.mask
            ffn_mask = layer.output.dense.mask
            self.reg_z_groups.append((
                (layer_mask0, qk_mask),
                (layer_mask0, qk_mask),
                (layer_mask0, vo_mask),
                (vo_mask, layer_mask1),
                (layer_mask1, ffn_mask),
                (ffn_mask, layer_mask2),
            ))
            past_layer_norm = layer.output.LayerNorm
    
    def set_reg_params_state(self, activate: bool):
        for name, module in self.model.named_modules():
            if isinstance(module, Mask):
                module.set_state(activate)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n not in self.reg_params)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n not in self.reg_params)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in self.reg_params and "reg" not in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.reg_learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in self.reg_params and "reg" in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": -self.args.reg_learning_rate,
                }
            ]
            
            optimizer_cls, optimizer_kwargs = DefaultTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    
    def train(self, 
        resume_from_checkpoint: Optional[Union[str, bool]] = None, 
        trial: Union["optuna.Trial", Dict[str, Any]] = None, 
        ignore_keys_for_eval: Optional[List[str]] = None, 
        **kwargs
    ):
        self.distill_switch = True
        result = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        self.distill_switch = False
        return result


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs, output_hidden_states=True)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # Distill Loss
        if self.distill_switch:
            assert isinstance(outputs, dict)
            alpha_1 = self.args.distill_alpha_1
            distill_loss = self.compute_distill_loss(
                unwrap_model(model),
                inputs, 
                outputs["logits"], 
                outputs["hidden_states"]
            )
            loss = (1 - alpha_1) * loss + distill_loss
        
        # Lagrangian Loss
        if self.reg_switch:
            lagrangian_loss = self.compute_lagrangian_loss()
            loss = loss + lagrangian_loss

        return (loss, outputs) if return_outputs else loss
    
    def mask_select(self, 
        value: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        assert value.shape[:-1] == mask.shape
        D = value.shape[-1]
        value = value.view(-1, D)
        mask = mask.view(-1).bool()
        return value[mask]
    
    def compute_distill_loss(self, 
        model: SModel,
        inputs: Dict,
        s_logits: torch.Tensor,
        s_hidden_states: torch.Tensor,
    ):
        with torch.no_grad():
            t_outputs = self.t_model(**inputs, output_hidden_states=True)
            t_logits = t_outputs["logits"]
            t_hidden_states = t_outputs["hidden_states"]

        mask: torch.Tensor = inputs["attention_mask"]
        D = s_logits.shape[-1]
        T = self.args.distill_T
        alpha_1 = self.args.distill_alpha_1
        alpha_2 = self.args.distill_alpha_2
        
        pred_loss = self.kl_loss(
            torch.log_softmax(s_logits / T, dim=-1),
            torch.log_softmax(t_logits / T, dim=-1),
        )
        
        assert len(t_hidden_states) % len(s_hidden_states) == 0

        _layer_loss = []
        step_size = len(t_hidden_states) // len(s_hidden_states)
        proj = model.bert.distill_projection
        for t_h, s_h in zip(
            t_hidden_states[step_size-1::step_size], 
            s_hidden_states
        ):
            _layer_loss.append(
                self.mse_loss(
                    self.mask_select(t_h, mask),
                    proj(self.mask_select(s_h, mask))
                )
            )
        layer_loss = torch.stack(_layer_loss).sum()
        
        distill_loss = alpha_1 * pred_loss + alpha_2 * layer_loss
        
        return distill_loss

    
    def compute_target_sparsity(self):
        start_sparsity = 1.0
        target_sparsity = self.args.target_sparsity
        t_bar = self.reg_warmup_percent * (target_sparsity - start_sparsity) + start_sparsity
        return t_bar


    def compute_lagrangian_loss(self):
        if self.reg_switch:
            num_layers = self.model.config.num_hidden_layers
            hidden_size = self.model.config.hidden_size
            ffn_size = self.model.config.intermediate_size
            M = (hidden_size * hidden_size * 4 + hidden_size * ffn_size * 2) * num_layers
            lambda_1 = self.model.bert.reg_lambda_1
            lambda_2 = self.model.bert.reg_lambda_2
            params = []
            for mask_group in self.reg_z_groups:
                for in_mask, out_mask in mask_group:
                    params.append(torch.outer(in_mask.L(), out_mask.L()).sum())
            s = torch.stack(params).sum() / M
            t = self.compute_target_sparsity()
            
            lagrangian_loss = lambda_1 * (s - t) + lambda_2 * torch.pow(s - t, 2.)
            return lagrangian_loss
        else:
            return None

    @torch.no_grad()
    def compute_sparsity(self):
        num_layers = self.model.config.num_hidden_layers
        hidden_size = self.model.config.hidden_size
        ffn_size = self.model.config.intermediate_size
        M = (hidden_size * hidden_size * 4 + hidden_size * ffn_size * 2) * num_layers
        params = []
        for mask_group in self.reg_z_groups:
            for in_mask, out_mask in mask_group:
                params.append(torch.outer(in_mask.L(), out_mask.L()).sum())
        s = torch.stack(params).sum() / M
        return s

    @torch.no_grad()
    def compute_per_layer_sparsity(self):
        model: SModel = self.model
        sparsities = []
        past_layer_norm = model.bert.embeddings.LayerNorm
        sparsities.append(past_layer_norm.mask.L().sum().item() / past_layer_norm.mask.features)
        for layer in model.bert.encoder.layer:
            attn_norm = layer.attention.output.LayerNorm
            ffn_norm = layer.output.LayerNorm
            sparsities.append(attn_norm.mask.L().sum().item() / attn_norm.mask.features)
            sparsities.append(ffn_norm.mask.L().sum().item() / ffn_norm.mask.features)
        return sparsities

    def evaluate(self, 
        eval_dataset: Optional[Dataset] = None, 
        ignore_keys: Optional[List[str]] = None, 
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        with torch.no_grad():
            lambda_1 = self.model.bert.reg_lambda_1
            lambda_2 = self.model.bert.reg_lambda_2
            sparsity = self.compute_sparsity()
            t_sparsity = self.compute_target_sparsity()
            per_layer_sparsity = self.compute_per_layer_sparsity()
            lagrangian_loss = self.compute_lagrangian_loss()
            logger.info("lambda-1: {}".format(lambda_1))
            logger.info("lambda-2: {}".format(lambda_2))
            logger.info("sparsity = {}".format(sparsity))
            logger.info("t_sparsity = {}".format(t_sparsity))
            logger.info("per_layer_sparsity = {}".format(per_layer_sparsity))
            logger.info("lagrangian_loss = {}".format(lagrangian_loss))
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

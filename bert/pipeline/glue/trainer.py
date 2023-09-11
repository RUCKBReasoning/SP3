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
        self.structural_pruning_start_epoch = self.args.structural_pruning_start_epoch
        self.pruning_warmup_epoch = self.args.pruning_warmup_epoch
        self.pruning_steps = 0
        self.pruning_steps_stage2 = 0
        self.pruning_warmup_steps = None

    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        train_dataloader = self.trainer.get_train_dataloader()
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        self.pruning_warmup_steps = self.pruning_warmup_epoch * num_update_steps_per_epoch       

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch >= self.pruning_start_epoch \
            and state.epoch < self.structural_pruning_start_epoch:
            if self.trainer.reg_switch is False:
                self.trainer.reg_switch = True
                self.trainer.set_reg_params_state(True)
        elif state.epoch >= self.structural_pruning_start_epoch:
            if self.trainer.structural_switch is False:
                self.trainer.structural_switch = True
                self.trainer.start_sparsity = self.args.target_sparsity
                self.trainer.target_sparsity = self.args.structural_target_sparsity
                self.trainer.set_reg_structural_params_state(True)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch >= self.pruning_start_epoch \
            and state.epoch < self.structural_pruning_start_epoch:
            self.pruning_steps += 1
            self.trainer.reg_warmup_percent = min(1.0, self.pruning_steps / self.pruning_warmup_steps)
        elif state.epoch >= self.structural_pruning_start_epoch:
            self.pruning_steps_stage2 += 1
            self.trainer.reg_warmup_percent = min(1.0, self.pruning_steps_stage2 / self.pruning_warmup_steps)


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

        self.start_sparsity = 1.
        self.target_sparsity = self.args.target_sparsity

        self.reg_switch = False
        self.structural_switch = False
        self.reg_warmup_percent = 0.
        self.reg_params = []
        self.reg_z_groups = []
        self.init_reg_params()
        self.set_reg_params_state(False)
        self.set_reg_structural_params_state(False)
        
    def init_reg_params(self):
        for name, _ in self.model.named_parameters():
            if name.endswith('reg_lambda_1') or \
                name.endswith('reg_lambda_2') or \
                name.endswith('log_alpha'):
                self.reg_params.append(name)
        model: SModel = self.model
        past_layer_norm = model.bert.embeddings.LayerNorm
        for layer in model.bert.encoder.layer:
            hidden_mask0 = past_layer_norm.mask
            hidden_mask1 = layer.attention.output.LayerNorm.mask
            hidden_mask2 = layer.output.LayerNorm.mask
            qk_mask = layer.attention.self.query.mask
            vo_mask = layer.attention.self.value.mask
            filter_mask = layer.output.dense.mask
            MHA_mask = layer.attention.output.mask
            head_mask = layer.attention.self.mask
            FFN_mask = layer.output.mask
            self.reg_z_groups.append((
                hidden_mask0,
                hidden_mask1,
                hidden_mask2,
                qk_mask,
                vo_mask,
                filter_mask,
                MHA_mask,
                head_mask,
                FFN_mask,
            ))
            past_layer_norm = layer.output.LayerNorm
    
    def set_reg_params_state(self, activate: bool):
        for name, module in self.model.named_modules():
            if isinstance(module, Mask) and module.features > self.model.config.num_attention_heads:
                module.set_state(activate)
    
    def set_reg_structural_params_state(self, activate: bool):
        for name, module in self.model.named_modules():
            if isinstance(module, Mask) and module.features <= self.model.config.num_attention_heads:
                module.set_state(activate)
                print("set {} with state: [{}]".format(name, activate))

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
        
        assert len(t_hidden_states) == len(s_hidden_states)
        
        proj = model.bert.distill_projection
        t_hidden_states = [self.mask_select(t_h, mask) for t_h in t_hidden_states]
        s_hidden_states = [proj(self.mask_select(s_h, mask)) for s_h in s_hidden_states]
        
        match_index = []
        with torch.no_grad():
            T = torch.stack(t_hidden_states).unsqueeze(0)
            S = torch.stack(s_hidden_states).unsqueeze(1)
            dist = (T - S).pow(2.).mean(-1).mean(-1)  # dist[i, j] = || S_i - T_j ||
            assert len(dist.shape) == 2
        
        num_layers = len(s_hidden_states)
        for i in range(num_layers):
            match_index.append(dist[i, i:].argmin().item() + i)
        
        _layer_loss = []
        for i, s_h in enumerate(s_hidden_states):
            t_h = t_hidden_states[match_index[i]]
            _layer_loss.append(self.mse_loss(t_h, s_h))
        layer_loss = torch.stack(_layer_loss).sum()
        
        distill_loss = alpha_1 * pred_loss + alpha_2 * layer_loss
        
        return distill_loss
    
    def compute_target_sparsity(self):
        start_sparsity = self.start_sparsity
        target_sparsity = self.target_sparsity
        t_bar = self.reg_warmup_percent * (target_sparsity - start_sparsity) + start_sparsity
        return t_bar

    def compute_lagrangian_loss(self):
        if self.reg_switch:
            s = self.compute_sparsity()
            t = self.compute_target_sparsity()

            lambda_1 = self.model.bert.reg_lambda_1
            lambda_2 = self.model.bert.reg_lambda_2
            lagrangian_loss = lambda_1 * (s - t) + lambda_2 * torch.pow(s - t, 2.)
            return lagrangian_loss
        else:
            return None

    def compute_sparsity(self):
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        hidden_size = self.model.config.hidden_size
        ffn_size = self.model.config.intermediate_size
        M = (hidden_size * hidden_size * 4 + hidden_size * ffn_size * 2) * num_layers
        params = []
        for mask_group in self.reg_z_groups:
            hidden_mask0, \
            hidden_mask1, \
            hidden_mask2, \
            qk_mask, \
            vo_mask, \
            filter_mask, \
            MHA_mask, \
            head_mask, \
            FFN_mask = mask_group

            MHA_mask_L = MHA_mask.L() if self.structural_switch \
                else torch.tensor([1.0]).type_as(MHA_mask.log_alpha)
            head_mask_L = head_mask.L() if self.structural_switch \
                else torch.tensor([1.0]).type_as(head_mask.log_alpha)
            # head_mask_L = torch.tensor([1.0]).type_as(head_mask.log_alpha)
            FFN_mask_L = FFN_mask.L() if self.structural_switch \
                else torch.tensor([1.0]).type_as(FFN_mask.log_alpha)
            for in_mask, out_mask in (
                (hidden_mask0, qk_mask),
                (hidden_mask0, qk_mask),
                (hidden_mask0, vo_mask),
                (hidden_mask1, vo_mask),
            ):
                H = in_mask.features
                W = out_mask.features
                mask = torch.outer(in_mask.L(), out_mask.L())
                mask = mask.reshape(H, num_heads, W // num_heads)
                mask = mask * head_mask_L[None, :, None]
                mask = mask * MHA_mask_L
                params.append(mask.sum())
            for in_mask, out_mask in (
                (hidden_mask1, filter_mask),
                (hidden_mask2, filter_mask),
            ):
                params.append((torch.outer(in_mask.L(), out_mask.L()) * FFN_mask_L).sum())
        s = torch.stack(params).sum() / M
        return s

    @torch.no_grad()
    def compute_per_layer_sparsity(self):
        model: SModel = self.model
        h_sparsities = []
        b_sparsities = []
        past_layer_norm = model.bert.embeddings.LayerNorm
        h_sparsities.append(past_layer_norm.mask.L().sum().item() / past_layer_norm.mask.features)
        for layer in model.bert.encoder.layer:
            attn_norm = layer.attention.output.LayerNorm
            ffn_norm = layer.output.LayerNorm
            h_sparsities.append(attn_norm.mask.L().sum().item() / attn_norm.mask.features)
            h_sparsities.append(ffn_norm.mask.L().sum().item() / ffn_norm.mask.features)
        for mask_group in self.reg_z_groups:
            MHA_mask, \
            head_mask, \
            FFN_mask = mask_group[-3:]
            b_sparsities.append((
                MHA_mask.L().sum().item() / MHA_mask.features,
                head_mask.L().sum().item() / head_mask.features,
                FFN_mask.L().sum().item() / FFN_mask.features,
            ))
        return h_sparsities, b_sparsities

    def evaluate(self, 
        eval_dataset: Optional[Dataset] = None, 
        ignore_keys: Optional[List[str]] = None, 
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        with torch.no_grad():
            lambda_1 = self.model.bert.reg_lambda_1.item()
            lambda_2 = self.model.bert.reg_lambda_2.item()
            sparsity = self.compute_sparsity()
            t_sparsity = self.compute_target_sparsity()
            per_layer_h_sparsity, per_layer_b_sparsity = self.compute_per_layer_sparsity()
            lagrangian_loss = self.compute_lagrangian_loss()
            logger.info("lambda-1: {}".format(lambda_1))
            logger.info("lambda-2: {}".format(lambda_2))
            logger.info("sparsity = {}".format(sparsity))
            logger.info("t_sparsity = {}".format(t_sparsity))
            logger.info("per_layer_h_sparsity = {}".format(per_layer_h_sparsity))
            logger.info("per_layer_b_sparsity = {}".format(per_layer_b_sparsity))
            logger.info("lagrangian_loss = {}".format(lagrangian_loss))
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

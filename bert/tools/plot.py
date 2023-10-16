import numpy as np
from matplotlib import pyplot as plt

import os
import torch
from modeling.modeling_packed_bert import (
    PackedBertConfig,
    PackedBertLayerConfig,
    PackedBertForSequenceClassification as PModel
)
from typing import List

font = {
    "size": 20
}

def draw_figure_inter(p_config, name, dataset_name):
    per_layer_config: List[PackedBertLayerConfig] = p_config.per_layer_config
    hidden_size = p_config.hidden_size

    hidden_dim = []
    for layer in per_layer_config:
        hidden_dim.extend([
            layer.attn_output_dim,
            layer.ffn_output_dim,
        ])

    x_values = tuple(i for i in range(1, 13))
    group_y_values = {
        "$d$ of the MHA block": [x / hidden_size * 100 for x in hidden_dim][0::2],
        "$d$ of the FFN block": [x / hidden_size * 100 for x in hidden_dim][1::2],
    }

    x = np.arange(len(x_values))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in group_y_values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Sparsity (%)', fontdict=font)
    ax.set_xlabel('# Layers', fontdict=font)
    ax.set_xticks(x + width / 2.0, x_values)
    ax.legend(loc='upper right', ncols=1, fontsize=16)
    ax.set_ylim(0, 100)

    plt.savefig("tools/figures/{}/{}.png".format(dataset_name, name))
    plt.close()


def draw_figure_intra(p_config, name, dataset_name):
    per_layer_config: List[PackedBertLayerConfig] = p_config.per_layer_config
    hidden_size = p_config.hidden_size
    ffn_size = p_config.intermediate_size

    x_values = tuple(i for i in range(1, 13))
    group_y_values = {
        r"$d_{\rm QK}$": tuple(
            layer.qk_dim / hidden_size * 100 * (0 if layer.prune_attn else 1)
                for layer in per_layer_config
        ),
        r"$d_{\rm VO}$": tuple(
            layer.vo_dim / hidden_size * 100 * (0 if layer.prune_attn else 1)
                for layer in per_layer_config
        ),
        r"$d_{f}$": tuple(
            layer.ffn_dim / ffn_size * 100 * (0 if layer.prune_ffn else 1)
                for layer in per_layer_config
        ),
    }

    x = np.arange(len(x_values))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in group_y_values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Sparsity (%)', fontdict=font)
    ax.set_xlabel('# Layers', fontdict=font)
    ax.set_xticks(x + width, x_values)
    ax.legend(loc='upper right', ncols=3, fontsize=16)
    ax.set_ylim(0, 25)

    plt.savefig("tools/figures/{}/{}.png".format(dataset_name, name))
    plt.close()


def draw_figure_heads(p_config, name, dataset_name):
    per_layer_config: List[PackedBertLayerConfig] = p_config.per_layer_config
    num_heads = p_config.num_attention_heads

    hidden_dim = []
    for layer in per_layer_config:
        hidden_dim.append(layer.num_heads * (0 if layer.prune_attn else 1))

    x_values = tuple(i for i in range(1, 13))
    group_y_values = {
        "number of attention heads": [
            x / num_heads * 100 
                for x in hidden_dim
        ],
    }

    x = np.arange(len(x_values))  # the label locations
    width = 0.5  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in group_y_values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Sparsity (%)', fontdict=font)
    ax.set_xlabel('# Layers', fontdict=font)
    ax.set_xticks(x, x_values)
    ax.set_ylim(0, 100)

    plt.savefig("tools/figures/{}/{}.png".format(dataset_name, name))
    plt.close()


def run_glue():
    for dataset_name in ["mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]:
        s_output_dir = "/data2/hyx/projects/tcsp_v2/cache/checkpoints/bert/[glue]-[{}]-[0.06]".format(dataset_name)
        p_model_config_path = os.path.join(s_output_dir, "pmodel_config.bin")
        p_config: PackedBertConfig = torch.load(p_model_config_path)

        dir_path = "tools/figures/{}".format(dataset_name)
        os.makedirs(dir_path, exist_ok=True)

        draw_figure_inter(p_config, "exp-fig1", dataset_name)
        draw_figure_intra(p_config, "exp-fig2", dataset_name)
        draw_figure_heads(p_config, "exp-fig3", dataset_name)


def run_squad():
    dataset_name = "squad"
    s_output_dir = "/data2/hyx/projects/tcsp_v2/cache/checkpoints/bert/[squad]-[0.06]"
    p_model_config_path = os.path.join(s_output_dir, "pmodel_config.bin")
    p_config: PackedBertConfig = torch.load(p_model_config_path)

    dir_path = "tools/figures/squad"
    os.makedirs(dir_path, exist_ok=True)

    draw_figure_inter(p_config, "exp-fig1", dataset_name)
    draw_figure_intra(p_config, "exp-fig2", dataset_name)
    draw_figure_heads(p_config, "exp-fig3", dataset_name)


run_glue()
run_squad()

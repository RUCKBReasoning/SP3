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
    "size": 17
}

def draw_figure1(p_config, name, dataset_name):
    per_layer_config: List[PackedBertLayerConfig] = p_config.per_layer_config
    hidden_size = p_config.hidden_size
    ffn_size = p_config.intermediate_size

    x_values = tuple(i for i in range(1, 13))
    group_y_values = {
        "qk": tuple(
            layer.qk_dim / hidden_size * 100 * (0 if layer.prune_attn else 1)
                for layer in per_layer_config
        ),
        "vo": tuple(
            layer.vo_dim / hidden_size * 100 * (0 if layer.prune_attn else 1)
                for layer in per_layer_config
        ),
        "ffn": tuple(
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

    ax.set_ylabel('percentage (%)', fontdict=font)
    ax.set_title('Intermediate dimensions of each layer in PLM', fontdict=font)
    ax.set_xticks(x + width, x_values)
    ax.legend(loc='upper right', ncols=3, fontsize=16)
    ax.set_ylim(0, 25)

    plt.savefig("tools/figures/{}/{}.png".format(dataset_name, name))
    plt.close()

def draw_figure2(p_config, name, dataset_name):
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

    ax.set_ylabel('percentage (%)', fontdict=font)
    ax.set_title('Number of attention heads of each layer in PLM', fontdict=font)
    ax.set_xticks(x, x_values)
    ax.legend(loc='upper right', ncols=3, fontsize=16)
    ax.set_ylim(0, 100)

    plt.savefig("tools/figures/{}/{}.png".format(dataset_name, name))
    plt.close()


def draw_figure3(p_config, name, dataset_name):
    per_layer_config: List[PackedBertLayerConfig] = p_config.per_layer_config
    hidden_size = p_config.hidden_size

    hidden_dim = []
    for layer in per_layer_config:
        hidden_dim.extend([
            layer.input_dim,
            layer.attn_output_dim
        ])
    hidden_dim.append(per_layer_config[-1].ffn_output_dim)

    x_values = tuple(i for i in range(0, 25))
    group_y_values = {
        "hidden_dim": [x / hidden_size * 100 for x in hidden_dim],
    }

    x = np.arange(len(x_values))  # the label locations
    width = 0.5  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in group_y_values.items():
        offset = width * multiplier
        rects = ax.plot(x + offset, measurement, label=attribute, linewidth=2)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('percentage (%)', fontdict=font)
    ax.set_title('Hidden dimensions of each layer in PLM', fontdict=font)
    ax.set_xticks(x, x_values)
    ax.legend(loc='upper right', ncols=3, fontsize=16)
    ax.set_ylim(0, 100)

    plt.savefig("tools/figures/{}/{}.png".format(dataset_name, name))
    plt.close()


def run():
    dataset_name = "mrpc"
    s_output_dir = "/data2/hyx/projects/tcsp_v2/cache/checkpoints/bert/[glue]-[{}]-[0.06]".format(dataset_name)
    p_model_config_path = os.path.join(s_output_dir, "pmodel_config.bin")
    p_config: PackedBertConfig = torch.load(p_model_config_path)

    draw_figure1(p_config, "exp-fig1", dataset_name)
    draw_figure2(p_config, "exp-fig2", dataset_name)
    draw_figure3(p_config, "exp-fig3", dataset_name)


run()

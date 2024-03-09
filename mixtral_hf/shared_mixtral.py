from typing import List, Optional, Tuple, Union

from torch import nn
import torch
from torch.nn import CrossEntropyLoss

from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM, 
    MixtralModel,
    MixtralDecoderLayer, 
    MixtralPreTrainedModel
)

class SharedMixtral(MixtralModel):
    def __init__(self, config):
        super().__init__(config)
        # num_layer = config.num_hidden_layers // 2
        # modules = [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(num_layer)] * 2
        modules = [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]

        self.layers = nn.ModuleList(modules)

class SharedMixtralForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = SharedMixtral(config)


def format_number(num):
    if abs(num) >= 10**12:  # Trillion
        return "{:.2f}T".format(num / 10**12)
    elif abs(num) >= 10**9:  # Billion
        return "{:.2f}B".format(num / 10**9)
    elif abs(num) >= 10**6:  # Million
        return "{:.2f}M".format(num / 10**6)
    else:
        return str(num)    
def show_total_params(model):
    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])    
    print('trainable params: ', format_number(params))

if __name__ =='__main__':
    from  mixtral import MixtralConfig_HF
    config = MixtralConfig_HF.from_name("debug")
    print('config', config._attn_implementation)
    model = SharedMixtralForCausalLM(config)
    show_total_params(model)
    # input_ids=torch.tensor([[1,2,3,4]])
    # labels=torch.tensor([[1,2,3,4]])
    # model(input_ids=input_ids, labels=labels)
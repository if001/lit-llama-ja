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
        num_layer = config.num_hidden_layers // 2
        
        # modules = [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(num_layer)] * 2
        modules = [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(num_layer)]
        self.layers = nn.ModuleList(modules)

class SharedMixtralForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = SharedMixtral(config)

# from  mixtral import MixtralConfig_HF
# config = MixtralConfig_HF.from_name("Mixtral-300M-llm-jp-tk")
# model = SharedMixtralForCausalLM(config)
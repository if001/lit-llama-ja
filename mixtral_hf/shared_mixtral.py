from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralModel,MixtralDecoderLayer
from torch import nn

class SharedMixtral(MixtralModel):
    def __init__(self, config):
        super().__init__(config)
        num_layer = config.num_hidden_layers // 2
        modules = [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(num_layer)] * 2
        self.layers = nn.ModuleList(modules)

class SharedMixtralForCausalLM(SharedMixtral):
    def __init__(self, config):
        super().__init__(config)
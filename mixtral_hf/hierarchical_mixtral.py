from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
from torch import nn

class HierarchicalMixtral(MixtralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        
        self.rnn = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            batch_first = batch_first)    

    def forward(self):
        pass
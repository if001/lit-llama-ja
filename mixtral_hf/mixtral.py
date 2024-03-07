import json
from typing import Optional, Tuple, List
import torch

from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from transformers.models.mixtral.configuration_mixtral import MixtralConfig



class MixtralForCausalLM_HF(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    # @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()        
        return logits, outputs.router_logits

class MixtralConfig_HF(MixtralConfig):
    def __init__(self,                 
                vocab_size=35000,
                hidden_size=4096,
                intermediate_size=14336,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                num_experts_per_tok=2,
                num_local_experts=8,
                **kwargs):
        
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            num_experts_per_tok=num_experts_per_tok,
            num_local_experts=num_local_experts,
            output_router_logits=True,
            **kwargs)

    def save(self, output_dir):
        """
        Save member variables of this instance to a JSON file.

        Parameters:
        - output_dir: The output dir of the JSON file to save to.
        """        
        member_vars = {k: v for k, v in self.__dict__.items() if not callable(v)}
        
        output_file = f'{output_dir}/training_config.json'
        with open(output_file, 'w') as f:
            json.dump(member_vars, f, ensure_ascii=False, indent=4)
        print(f'save training config... {output_file}')

    def debug(self):
        print('='*100)
        print('print training config...')
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print('='*100)    
        
    @classmethod
    def from_name(cls, size='7B'):
        if size == "Mixtral-100M":
            conf = dict(
                hidden_size=640,
                intermediate_size=2400,
                num_hidden_layers=8,
                num_attention_heads=8,
                num_key_value_heads=4,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                num_experts_per_tok=2,
                num_local_experts=6,
            )
            return cls(**conf)
        if size == "Mixtral-100M-llm-jp-tk":
            conf = dict(
                vocab_size=50570,
                hidden_size=640,
                intermediate_size=2400,
                num_hidden_layers=8,
                num_attention_heads=8,
                num_key_value_heads=4,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=4,
                num_experts_per_tok=2,
                num_local_experts=6,
                max_position_embeddings=4096
            )
            return cls(**conf)
        if size == "Mixtral-700M-llm-jp-tk":
            conf = dict(
                vocab_size=50570,
                hidden_size=2048,
                intermediate_size=2000,
                num_hidden_layers=6,
                num_attention_heads=8,
                num_key_value_heads=4,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=4,
                num_experts_per_tok=2,
                num_local_experts=6,
                max_position_embeddings=2048
            )
            return cls(**conf)
        if size == "Mixtral-300M-llm-jp-tk":
            conf = dict(
                vocab_size=50570,
                hidden_size=1024,
                intermediate_size=2400,
                num_hidden_layers=10,
                num_attention_heads=6,
                num_key_value_heads=2,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=4,
                num_experts_per_tok=2,
                num_local_experts=6,
                max_position_embeddings=1024
            )
            return cls(**conf) 
        raise ValueError("")
        

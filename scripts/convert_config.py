## lit_llama config to transformers config
import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from transformers import LlamaConfig

from lit_llama.config_llama2 import Llama2Config

# LlamaConfig {
#   "attention_bias": false,
#   "bos_token_id": 1,
#   "eos_token_id": 2,
#   "hidden_act": "silu",
#   "hidden_size": 4096,
#   "initializer_range": 0.02,
#   "intermediate_size": 11008,
#   "max_position_embeddings": 2048,
#   "model_type": "llama",
#   "num_attention_heads": 32,
#   "num_hidden_layers": 32,
#   "num_key_value_heads": 32,
#   "pretraining_tp": 1,
#   "rms_norm_eps": 1e-06,
#   "rope_scaling": null,
#   "rope_theta": 10000.0,
#   "tie_word_embeddings": false,
#   "transformers_version": "4.34.0",
#   "use_cache": true,
#   "vocab_size": 32000
# }

def convert_config(model_size):
    t_config = LlamaConfig()
    config = Llama2Config.from_name(model_size)
    print(t_config)
    print('='*100)
    t_config.hidden_size = config.n_embd
    t_config.max_position_embeddings = config.block_size
    t_config.intermediate_size = config.intermediate_size

    t_config.num_attention_heads = config.n_head
    t_config.num_hidden_layers = config.n_head
    t_config.num_key_value_heads = config.n_head
    t_config.rms_norm_eps = config.norm_eps

    ## tokenizer config
    t_config.bos_token_id = 1
    t_config.eos_token_id = 2
    t_config.vocab_size = 35000
    print(t_config)
    print('='*100)
    return t_config
    
def main(model_size: str = ""):
    convert_config(model_size)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main, as_positional=False)
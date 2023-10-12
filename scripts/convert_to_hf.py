## convert_hf_checkpoint_for_llama2.pyで
## layer名を変更したあとのweightをhuggingfaceように変換する

from convert_config import convert_config
from transformers import AutoModelForCausalLM
import torch

def main(save_dir: str = "", model_size: str = "", weight_path: str = ""):        
    t_config = convert_config(model_size)
    model = AutoModelForCausalLM.from_config(t_config)
    for k in model.state_dict():
        print(k)

    pytorch_weights = torch.load(weight_path)
    model.load_state_dict(pytorch_weights)

    # model.save_pretrained(save_dir=save_dir)
    model.save_pretrained(save_dir, push_to_hub=True, repo_name="if001/llama2_ja_small")

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
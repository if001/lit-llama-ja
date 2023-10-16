## convert_hf_checkpoint_for_llama2.pyでlayer名を変更したあとのweightをhuggingface用に変換
## huggingfaceへupload

from convert_config import convert_config
from transformers import AutoModelForCausalLM
import torch

def main(model_size: str = "", weight_path: str = "", repo_id: str = ""):
    t_config = convert_config(model_size)
    model = AutoModelForCausalLM.from_config(t_config)
    for k in model.state_dict():
        print(k)

    pytorch_weights = torch.load(weight_path)
    model.load_state_dict(pytorch_weights)

    # model.save_pretrained(save_dir=save_dir)
    # model.save_pretrained(save_dir, push_to_hub=True, repo_name="if001/llama2_ja_small")
    model.push_to_hub(repo_id)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
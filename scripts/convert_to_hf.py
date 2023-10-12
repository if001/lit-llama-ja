## convert_hf_checkpoint_for_llama2.pyで
## layer名を変更したあとのweightをhuggingfaceように変換する

from convert_config import convert_config
from transformers import AutoModel
import torch

def main(save_dir: str = "", model_size: str = "", weight_path: str = ""):        
    t_config = convert_config(model_size)
    model = AutoModel.from_config(t_config)
    for k, v in model.state_dict():
        print(k, v.size())

    pytorch_weights = torch.load(weight_path)
    model.load_state_dict(pytorch_weights)

    model.save_pretrained(save_dir=save_dir)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
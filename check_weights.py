import torch
from lit_llama.model_llama2 import GPT
from lit_llama.config_llama2 import Llama2Config

from lightning.fabric.utilities.load import _lazy_load

def main(
        model_size: str = "7B",
        checkpoint_file: str = "./",
):
    print('load model: ', model_size)
    print('checkpoint_file: ', checkpoint_file)

    config = Llama2Config.from_name(model_size)

    model = GPT(config)

    compiled_model = torch.compile(model)
    state_dict = _lazy_load(checkpoint_file)
    compiled_model.load_state_dict(state_dict, strict=True)
    print('-'*100)
    for v in model.state_dict():
        print(v)
    print('-'*100)


if __name__ == '__main__':
    from jsonargparse.cli import CLI

    CLI(main)    
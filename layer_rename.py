import torch
from lit_llama.model_llama2 import GPT
from lit_llama.config_llama2 import Llama2Config

from lightning.fabric.utilities.load import _lazy_load

def main(
        model_size: str,
        checkpoint_file: str,
        save_file: str,
):
    print('load model: ', model_size)
    print('checkpoint_file: ', checkpoint_file)
    print('save_file: ', save_file)

    config = Llama2Config.from_name(model_size)

    model = GPT(config)

    compiled_model = torch.compile(model)
    state_dict = _lazy_load(checkpoint_file)
    compiled_model.load_state_dict(state_dict, strict=True)


    original_names = list(model.state_dict().keys())
    compiled_names = list(compiled_model.state_dict().keys())

    name_mapping = dict(zip(compiled_names, original_names))

    new_state_dict = {}
    for compiled_name, param in compiled_model.state_dict().items():
        new_name = name_mapping[compiled_name]
        new_state_dict[new_name] = param

    model.load_state_dict(new_state_dict)
    
    torch.save(model.state_dict(), save_file)


if __name__ == '__main__':
    from jsonargparse.cli import CLI

    CLI(main)    
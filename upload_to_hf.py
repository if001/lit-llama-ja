import sys
import warnings
from pathlib import Path
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import HFTokenizer
from lit_llama.utils import lazy_load, llama_model_lookup, quantization
from mixtral_hf.mixtral import MixtralConfig_HF

from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

def main(
    repo_id: str = "",
    checkpoint_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),    
    model_name: str = "7B",
) -> None:  
    assert checkpoint_path.is_file(), checkpoint_path    

    print(f"Loading model ...{model_name}", file=sys.stderr)
    with lazy_load(checkpoint_path) as checkpoint:
        # name = llama_model_lookup(checkpoint)        
        config = MixtralConfig_HF.from_name(model_name)
        # model = MixtralForCausalLM_HF(config)
        model = MixtralForCausalLM(config)
        model.load_state_dict(checkpoint)

    model.push_to_hub(repo_id)

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore", 
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    CLI(main)

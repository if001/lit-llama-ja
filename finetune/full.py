"""
Instruction-tuning on the Alpaca dataset using a regular finetuning procedure (updating all layers).

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import sys
from pathlib import Path
import os
import time
from functools import partial

import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
# from lit_llama.model import Block, LLaMA, LLaMAConfig
# from lit_llama.tokenizer import Tokenizer

from lit_llama.utils import save_model_checkpoint, chunked_cross_entropy
from scripts.prepare_alpaca import generate_prompt, generate_prompt_ja
from lit_llama.config_llama2 import Llama2Config
from lit_llama.model_llama2 import GPT
from lit_llama.tokenizer import HFTokenizer

instruction_tuning = True
eval_interval = 1000
save_interval = 1000
eval_iters = 1000
log_interval = 5000
devices = 1

# Hyperparameters
# learning_rate = 3e-5
learning_rate = 3e-5
batch_size = 128 / devices
micro_batch_size = 4
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
# epoch_size = 50000  # train dataset size
epoch_size = 120000  # train dataset size
num_epochs = 10
max_iters = num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 0.0001
block_size = 2048
warmup_iters = 1000


def main(
    model_size: str = "7B",
    data_dir: str = "data/alpaca",
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    out_dir: str = "out/full/alpaca",
):

    # auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    # strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block, limit_all_gathers=True)
    log_dir = out_dir + '/logs'
    logger = TensorBoardLogger(log_dir, name="model")
    precision="16-mixed" ## for v100
    fabric = L.Fabric(accelerator="cuda", devices=devices, precision=precision, loggers=logger)
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

    config = Llama2Config.from_name(model_size)
    config.block_size = block_size

    checkpoint = torch.load(pretrained_path)

    with fabric.device:
        # torch.set_default_tensor_type(torch.HalfTensor)
        # model = GPT(config).bfloat16()
        model = GPT(config)
        # torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False) 

    model = fabric.setup_module(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, foreach=False, weight_decay=weight_decay)
    optimizer = fabric.setup_optimizers(optimizer)

    train(fabric, model, optimizer, train_data, val_data, out_dir)

    # Save the final checkpoint at the end of training
    save_model_checkpoint(fabric, model, os.path.join(out_dir, "lit-llama-full-finetuned.pth"))


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    out_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    model.train()

    for iter_num in range(max_iters):

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0

        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()
        
        input_ids, targets = get_batch(fabric, train_data)
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            # loss = loss_fn(logits, targets)
            loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)            
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data)
                print('-'*100)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")                
                print('-'*100)
                fabric.log_dict(
                    {"iter": iter_num, "val_loss": val_loss, "step": step_count, "lr": lr}, step=iter_num
                )
                fabric.barrier()

            if step_count % save_interval == 0:
                output_file = os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth")
                print(f"Saving weights to {str(output_file)}")
                save_model_checkpoint(fabric, model, output_file)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
            fabric.log_dict(
                {"iter": iter_num, "train_loss": loss, "step": step_count, "lr": lr}, step=iter_num
            )


def generate_response(model, instruction):
    tokenizer = HFTokenizer("/content/drive/MyDrive/tokenizer/spm/tokenizer_35000.json")

    # sample = {"instruction": instruction, "input": ""}
    sample = {"instruction": instruction, "input": ""}
    prompt = instruction
    if instruction_tuning:
        prompt = generate_prompt_ja(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=block_size,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = "週末に見るべき映画を推薦し、その理由を説明してください。"
    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)

    output = generate_response(model, instruction)
    #fabric.print(instruction)
    fabric.print(output)

    model.train()
    return out.item()


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss


def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)

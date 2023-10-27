import os
import sys
import math
import glob
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Optional

import lightning as L
from lightning.fabric.strategies import FSDPStrategy, DeepSpeedStrategy
from lightning.fabric.utilities.load import _lazy_load
from lightning.pytorch.loggers import TensorBoardLogger

import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import numpy as np

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama.model import Block, LLaMA, LLaMAConfig
from lit_llama.packed_dataset import PackedDataset, CombinedDataset
from lit_llama.utils import save_model_checkpoint, save_model_checkpoint_with_fabric, chunked_cross_entropy

from lit_llama.model_llama2 import GPT
from lit_llama.config_llama2 import Llama2Config
from lit_llama.training_config import TrainingConfig


# out_dir = "out/training"
# save_interval = 1000
# eval_interval = 1000
# eval_iters = 100
# log_interval = 1

save_interval = 1000
save_interval = 100
save_interval = 50
eval_interval = 100
eval_interval = 50
eval_iters = 100
log_interval = 500

# compile = False

# # Hyperparameters for 7B
# learning_rate = 6e-4
# batch_size = 125
# micro_batch_size = 5
# max_iters = 600000  # num_epochs * (epoch_size // micro_batch_size) // devices
# weight_decay = 1e-1
# beta1 = 0.9
# beta2 = 0.95
# grad_clip = 1.0
# decay_lr = True
# warmup_iters = 2000
# lr_decay_iters = max_iters
# min_lr = 6e-5

# # # Hyperparameters for 49M
# learning_rate = 0.0008
# # batch_size = 125
# batch_size = 128
# # micro_batch_size = 5
# micro_batch_size = 4
# micro_batch_size = 2
# # max_iters = 80000  # num_epochs * (epoch_size // micro_batch_size) // devices
# max_iters = 143000  # num_epochs * (epoch_size // micro_batch_size) // devices
# weight_decay = 0.1
# beta1 = 0.9
# beta2 = 0.95
# grad_clip = 1.0
# decay_lr = True
# warmup_iters = 2000
# lr_decay_iters = max_iters
# min_lr = 0.00008

# ## for 49M
# learning_rate = 0.0009
# learning_rate = 0.0009
# min_lr = 0.00009
# batch_size = 128
# micro_batch_size = 2
# max_iters = 143000  # num_epochs * (epoch_size // micro_batch_size) // devices
# weight_decay = 0.1
# beta1 = 0.9
# beta2 = 0.95
# grad_clip = 1.0
# decay_lr = True
# warmup_iters = 2000
# lr_decay_iters = max_iters


# ## for 125M

# ## for 350M
# learning_rate = 0.001
# learning_rate = 0.005
# ## learning_rate = 0.005
# min_lr = 0.00006
# batch_size = 128
# batch_size = 32
# micro_batch_size = 2
# max_iters = 143000  # num_epochs * (epoch_size // micro_batch_size) // devices
# ## weight_decay = 0.0001
# weight_decay = 0.001
# beta1 = 0.9
# beta2 = 0.95
# grad_clip = 2.0
# decay_lr = True
# warmup_iters = 1000
# lr_decay_iters = max_iters


# Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1
# data_config = [
#     ("arxiv", 2.5),
#     ("book", 4.5),
#     ("c4", 15.0),
#     ("cc", 67.0),
#     ("github", 4.5),
#     ("stackexchange", 2.0),
#     ("wikipedia", 4.5),
# ]
val_data_config = [
    # ("aozorabunko-clean-sin", 1.0),
    ("wikinews-ja-20230728", 1.0),
    ("wikinews-en-20230728", 1.0),
]
train_data_config = [ 
    ('wikipedia-ja-20230720', 1.0),
    ('wikipedia-en-20230720', 1.0),
    ('open-text-books', 1.0),
    ('oscar_2023_filtered', 1.0),
    ('aozorabunko-clean-sin',1.0)
]

## 日本語: 997.79M, 英語: 3.80B/3
train_data_config = [ 
    ('wikipedia-ja-20230720', 1.0),
    ('wikipedia-en-20230720', 0.3),
    ('open-text-books', 1.0),
    ('aozorabunko-clean-sin',1.0)
]


def format_number(num):
    if abs(num) >= 10**12:  # Trillion
        return "{:.2f}T".format(num / 10**12)
    elif abs(num) >= 10**9:  # Billion
        return "{:.2f}B".format(num / 10**9)
    elif abs(num) >= 10**6:  # Million
        return "{:.2f}M".format(num / 10**6)
    else:
        return str(num)
    
def show_total_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])    
    print('trainable params: ', format_number(params))

    
def main(
    devices: int = 4,
    train_data_dir: Path = "data/lit-redpajama",
    val_data_dir: Optional[Path] = None,
    model_size: str = "7B",
    out_dir: str = "out/training",
    load_dir: Optional[str] = None,
    restart_iter: int = 0,
    log_dir: str = "./logs"
) -> None:
    trainingConfig = TrainingConfig.from_name(model_size)
    trainingConfig.debug()
    trainingConfig.save(out_dir)

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Block}
    )
    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block, limit_all_gathers=True
    )
    strategy = DeepSpeedStrategy(stage=1, 
                                 zero_optimization=True,
                                 allgather_partitions=True, 
                                 allgather_bucket_size=500000000,
                                 overlap_comm=True,
                                 reduce_scatter=True,
                                 reduce_bucket_size=500000000,                                 
                                 contiguous_gradients=True,
                                 initial_scale_power=12,                                
                                 loss_scale=0,
                                 loss_scale_window=1000,
                                 hysteresis=2,
                                 min_loss_scale=0.5
                                 )
    # strategy = 'ddp'
    # fabric = L.Fabric(accelerator="cuda", devices=devices, precision="bf16-mixed", strategy=strategy)
    # fabric = L.Fabric(accelerator="cuda", devices=devices, precision="16-true", strategy=strategy)
    # fabric = L.Fabric(accelerator="cuda", devices=devices, precision="bf16-mixed", strategy=strategy, loggers=TensorBoardLogger(log_dir, name="model"))
    logger = TensorBoardLogger(log_dir, name="model")
    
    precision="16-mixed" ## for v100
    # precision="bf16-mixed" ## for A100
    fabric = L.Fabric(accelerator="cuda", devices=devices, precision=precision, loggers=logger)

    fabric.launch()
    fabric.seed_everything(1337)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    # config = LLaMAConfig.from_name("7B")
    config = Llama2Config.from_name(model_size)
    config.nef = False
    config.debug()
    config.save(out_dir)
    print('out_dir: ', out_dir)
    print('val data dir:', val_data_dir)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=trainingConfig.micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=1338,
    )
    if val_dataloader is None:
        print('val data is None...')
    
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    with fabric.device:
        # torch.set_default_dtype(torch.bfloat16)
        # torch.set_default_dtype(torch.float16)
        print('dtype: ', torch.get_default_dtype())
        model = GPT(config)
        ## model = LLaMA(config)
        model.apply(model._init_weights)
        # torch.set_default_dtype(torch.float32)
        if load_dir:
            print('load from checkpoint...', load_dir)
            state_dict = _lazy_load(load_dir)
            model.load_state_dict(state_dict, strict=True)
            # checkpoint = torch.load(load_dir)
            # model.load_state_dict(checkpoint)
            # fabric.load(load_dir, {"model": model, "optimizer": optimizer})

    # if compile:
    #     model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainingConfig.learning_rate,
        weight_decay=trainingConfig.weight_decay,
        betas=(trainingConfig.beta1, trainingConfig.beta2),
        foreach=False,
    )

    model, optimizer = fabric.setup(model, optimizer)       
    show_total_params(model)    

    process_batch_size = trainingConfig.batch_size // devices
    gradient_accumulation_iters = process_batch_size // trainingConfig.micro_batch_size    

    train(trainingConfig, fabric, model, optimizer, train_dataloader, val_dataloader, gradient_accumulation_iters, devices, out_dir, restart_iter)
    fabric.print(f"Saving checkpoint to {out_dir}")
    save_model_checkpoint_with_fabric(fabric, model, out_dir, f"iter-{trainingConfig.max_iters:06d}-ckpt.pth")
    logger.save()

def train(
    trainingConfig: TrainingConfig,
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    grad_accum_steps: int,
    devices: int,
    out_dir: str,
    restart_iter: int = 0,    
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """

    step_count = 0

    step_time = 0.0
    tokens = 0
    tokens_sec = 0.0
    prev_t1 = time.time()

    for iter_num, train_data in enumerate(train_dataloader):
        iter_num = iter_num + restart_iter
        t0 = time.time()

        # determine and set the learning rate for this iteration
        lr = trainingConfig.get_lr(iter_num) if trainingConfig.decay_lr else trainingConfig.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()
        
        is_accumulating = (iter_num + 1) % grad_accum_steps != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            # loss = torch.nn.functional.cross_entropy(
            #     logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            # )            
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / grad_accum_steps)

        t1 = time.time()

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=trainingConfig.grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            t1 = time.time()

            if val_dataloader is not None and step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_dataloader)
                print('-'*100)
                fabric.print(f"iter: {iter_num},  val loss: {val_loss:.4f}")
                print('-'*100)
                fabric.barrier()
                fabric.log_dict(
                    {"iter": iter_num, "val_loss": val_loss, "step": step_count, "lr": lr}, step=iter_num
                )
                ## fabric.loggers[0].save()

            if step_count % save_interval == 0:
                fabric.print("-"*200)
                fabric.print(f"Saving checkpoint to {out_dir} _ iter-{iter_num:06d}-ckpt.pth")
                fabric.print("-"*200)
                save_model_checkpoint_with_fabric(
                    fabric, model, out_dir, f"iter-{iter_num:06d}-ckpt.pth"
                )

        dt = t1 - t0

        tokens += trainingConfig.micro_batch_size * model.config.block_size
        step_time += t1 - prev_t1
        prev_t1 = t1

        if iter_num % log_interval == 0:
            # tokens_sec_str = f"{tokens / step_time:.0f}" if not is_accumulating else "-"

            fabric.print(
                    f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms, lr: {lr}, step_count: {step_count}"
            )
            fabric.log_dict(
                {"iter": iter_num, "train_loss": loss, "step": step_count, "lr": lr}, step=iter_num
            )
            fabric.log("loss_2", loss)
            # fabric.print(
            #         f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms, speed: {tokens_sec_str} toks/s/device"
            # )

        if not is_accumulating:
            tokens = 0
            step_time = 0.0

        if iter_num > trainingConfig.max_iters:
            break


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        # loss = torch.nn.functional.cross_entropy(
        #     logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        # )
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)        
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: str,
    fabric,
    data_config: str,
    shuffle: bool = True,
    seed: int = 12345,
) -> DataLoader:
    datasets = []
    for prefix, _ in data_config:
        filenames = glob.glob(os.path.join(data_dir, prefix + "*"))
        n_chunks = len(filenames)
        # n_chunks = 4 ## default
        dataset = PackedDataset(            
            filenames, n_chunks=n_chunks, block_size=block_size, shuffle=shuffle, seed=seed,
            num_processes=fabric.world_size, process_rank=fabric.global_rank, wrap=True
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: str = "data/lit-redpajama",
    val_data_dir: Optional[str] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        data_config=train_data_config,
        shuffle=True,
        seed=seed,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            data_config=val_data_config,
            shuffle=False,
            seed=seed,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)

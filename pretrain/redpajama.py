import os
import sys
import math
import glob
import time
import json
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

# save_interval = 1000
# save_interval = 100
# save_interval = 50
# save_interval = 25
# save_interval = 8192

# eval_interval = 100
# eval_interval = 50
# eval_interval = 25
# eval_interval = 8192
# eval_iters = 100
# log_interval = 500

# compile = False

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
# val_data_config = [
#     # ("aozorabunko-clean-sin", 1.0),
#     ("wikinews-ja-20230728", 1.0),
#     ("wikinews-en-20230728", 1.0),
# ]
# train_data_config = [ 
#     ('wikipedia-ja-20230720', 1.0),
#     ('wikipedia-en-20230720', 1.0),
#     ('open-text-books', 1.0),
#     ('oscar_2023_filtered', 1.0),
#     ('aozorabunko-clean-sin',1.0)
# ]
# ## 日本語: 997.79M, 英語: 3.80B/3
# train_data_config = [ 
#     ('wikipedia-ja-20230720', 1.0),
#     ('wikipedia-en-20230720', 0.3),
#     ('open-text-books', 1.0),
#     ('aozorabunko-clean-sin',1.0)
# ]


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

def reconnect_drive():
    from google.colab import drive
    drive._mount('/content/drive')
    print('reconnect to drive...')

def main(
    devices: int = 4,
    train_data_dir: Path = "data/lit-redpajama",
    val_data_dir: Optional[Path] = None,
    model_size: str = "7B",
    out_dir: str = "out/training",
    load_dir: Optional[str] = None,
    restart_iter: int = 0,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 0.001,
    interrupt: bool = False,
    train_data_rate: float = 1.0,
) -> None:    

    trainingConfig = TrainingConfig.from_name(model_size)
    if interrupt:
        print('interrupt setting!!!')
        trainingConfig.batch_size = batch_size
        trainingConfig.learning_rate = lr
        trainingConfig.min_lr = lr * 10
        trainingConfig.weight_decay = weight_decay
     
    out_dir_path = Path(out_dir) / f"model_{model_size}_b{trainingConfig.batch_size}_lr{trainingConfig.learning_rate}_wd{trainingConfig.weight_decay}"    
    out_model_dir = Path(out_dir_path / "models")    
    out_model_dir.mkdir(parents=True, exist_ok=True)
    print('out_model_dir: ', out_model_dir)    

    out_log_dir = Path(out_dir_path / "logs")    
    out_log_dir.mkdir(parents=True, exist_ok=True)
    print('out_log_dir: ', out_log_dir)

    trainingConfig.debug()
    trainingConfig.save(str(out_dir_path))

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

    logger = TensorBoardLogger(out_log_dir, name="model")
    
    precision="16-mixed" ## for v100
    # precision="bf16-mixed" ## for A100
    fabric = L.Fabric(accelerator="cuda", devices=devices, precision=precision, loggers=logger)

    fabric.launch()
    fabric.seed_everything(1337)

    if fabric.global_rank == 0:
        os.makedirs(str(out_dir_path), exist_ok=True)

    # config = LLaMAConfig.from_name("7B")
    config = Llama2Config.from_name(model_size)
    config.nef = False
    config.debug()
    config.save(str(out_dir_path))
    print('out_dir: ', str(out_dir_path))
    print('val data dir:', val_data_dir)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=trainingConfig.micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=1338,
        train_data_rate=train_data_rate
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
        print(model)
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

    ds_size = int(8e+9 * train_data_rate)
    print("ds size:", format_number(ds_size))
    train(trainingConfig, fabric, model, optimizer, train_dataloader, 
          val_dataloader, gradient_accumulation_iters, devices, str(out_model_dir), 
          restart_iter, interrupt, model_size, ds_size)
    fabric.print(f"Saving checkpoint to {str(out_model_dir)}")
    save_model_checkpoint_with_fabric(fabric, model, str(out_model_dir), f"iter-{trainingConfig.max_iters:06d}-ckpt.pth")
    try:
        logger.save()
    except Exception as e:
        print("error", e)

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
    interrupt: bool = False,
    model_size: str = "",
    ds_size: int = 8e+9
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """

    step_count = 0

    step_time = 0.0
    tokens = 0
    total_tokens = restart_iter * trainingConfig.micro_batch_size * model.config.block_size
    tokens_sec = 0.0
    prev_t1 = time.time()

    log_interval = 500
    log_interval = 1000
    eval_iters = 100
    # save_interval = 8192 / trainingConfig.batch_size
    # eval_interval = 8192 / trainingConfig.batch_size
    eval_interval = 12800 / trainingConfig.batch_size
    save_interval = 12800 / trainingConfig.batch_size
    # save_interval = 4096 / trainingConfig.batch_size
    # eval_interval = 4096 / trainingConfig.batch_size
    # save_interval = 500
    # eval_interval = 100
    total_time = 0

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
            # fabric.clip_gradients(model, optimizer, max_norm=trainingConfig.grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            t1 = time.time()

            if val_dataloader is not None and step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_dataloader, eval_iters=eval_iters)
                print('-'*100)
                fabric.print(f"iter: {iter_num},  val loss: {val_loss:.4f}")
                print('-'*100)
                fabric.barrier()
                l = {"iter": iter_num, "val_loss": val_loss, "step": step_count, "lr": lr}
                try:
                    fabric.log_dict(l, step=iter_num)
                except Exception as e:
                    print("error", e)
                    reconnect_drive()

                if interrupt:
                    print('interrupt!!')
                    output_file = f'{out_dir}/search_param-{model_size}-b{trainingConfig.batch_size}-lr{trainingConfig.learning_rate}-wb{trainingConfig.weight_decay}.json'
                    d = {"iter": iter_num, "step": step_count, "val_loss": f"{val_loss:.4f}", "loss": f"{loss.item():.4f}"}
                    with open(output_file, 'w') as f:
                        json.dump(d, f, ensure_ascii=False, indent=4)
                    print('save ', output_file)
                    break
                ## fabric.loggers[0].save()

            if step_count % save_interval == 0:
                fabric.print("-"*200)
                fabric.print(f"Saving checkpoint to {out_dir}/iter-{iter_num:06d}-ckpt.pth")
                fabric.print("-"*200)
                save_model_checkpoint_with_fabric(
                    fabric, model, out_dir, f"iter-{iter_num:06d}-ckpt.pth"
                )

        dt = t1 - t0
        total_time += dt

        tokens += trainingConfig.micro_batch_size * model.config.block_size
        total_tokens += trainingConfig.micro_batch_size * model.config.block_size
        step_time += t1 - prev_t1
        prev_t1 = t1

        if iter_num % log_interval == 0:
            # tokens_sec_str = f"{tokens / step_time:.0f}" if not is_accumulating else "-"

            _h = int(total_time // 3600)
            _m = int((total_time % 3600) // 60)
            _tokens_str = format_number(tokens)
            _total_tokens_str = format_number(total_tokens)
            _per = float(total_tokens*100/ds_size)
            fabric.print(
                    f"iter {iter_num}: loss {loss.item():.4f}, lr: {lr}, step_count: {step_count}, tokens: {_tokens_str}, total tokens: {_total_tokens_str}({_per:.2f}%), total time: {_h}h{_m:.2f}m"
            )

            try:
                fabric.log_dict(
                    {"iter": iter_num, "train_loss": loss, "step": step_count, "lr": lr}, step=iter_num
                )
            except Exception as e:
                print("error", e)
                reconnect_drive()      
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
    fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, eval_iters = 100
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
    train_data_rate: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    train_data_config = [ 
        ('wikipedia-ja-20230720', train_data_rate),
        ('wikipedia-en-20230720', train_data_rate),
        ('open-text-books', train_data_rate),
        ('oscar_2023_filtered', train_data_rate),
        ('aozorabunko-clean-sin',train_data_rate)
    ]
    val_data_config = [
        # ("aozorabunko-clean-sin", 1.0),
        ("wikinews-ja-20230728", train_data_rate),
        ("wikinews-en-20230728", train_data_rate),
    ]
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

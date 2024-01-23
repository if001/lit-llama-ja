from dataclasses import dataclass, field, concatenate_datasets
from typing import List, Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer, is_xpu_available

"""sample

!python finetune/sft_as_hf.py \
--model_name="if001/tiny_mixtral_ja" \
--tokenizer_name="if001/sentencepiece_ja" \
--dataset_name="izumi-lab/llm-japanese-dataset,kunishou/databricks-dolly-15k-ja,if001/oasst1_ja_ppl" \
--seq_length=2048 \
--logging_steps=1000 \
--save_total_limit=3 \
--save_steps=1000 \
--batch_size=32 \
--trust_remote_code=True \
--output_dir=""
"""



def format_instruction(ds):    
    input = ds['train']['input']
    instruction = ds['train']['instruction']
    output = ds['train']['output']
    if input is None:
        f"""以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい
### 指示:
{instruction}

### 応答:
{output}
"""

    return f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。
### 指示:
{instruction}

### 入力:
{input}

### 応答:
{output}
"""




# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    # report_to: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    # push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "Whether to activate fp16 mixed precision"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "Whether to activate bf16 mixed precision"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    # hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    mixed_precision: Optional[str] = field(default="bf16", metadata={"help": "Mixed precision training"})
    target_modules: Optional[List[str]] = field(default=None, metadata={"help": "Target modules for LoRA adapters"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the dataset
if script_args.dataset_name in ",":
    datasets = []
    dataset_names = script_args.dataset_name.split(",")    
    for name in dataset_names:
        ds = load_dataset(name, split="train")
        ds = ds.select(range(3))
        ds = ds.shuffle().map(format_instruction)
        print(ds)
        datasets.append(ds)

dataset = concatenate_datasets(datasets)
print('2', dataset)
dataset = dataset.train_test_split(test_size=0.3)
print('3', dataset)
exit(0)
# Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # Copy the model to each device
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    # report_to=script_args.report_to,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    # push_to_hub=script_args.push_to_hub,
    # hub_model_id=script_args.hub_model_id,
    gradient_checkpointing=script_args.gradient_checkpointing,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    # TODO: uncomment that on the next release
    # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
)

# Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=script_args.target_modules,
    )
else:
    peft_config = None

# Define the Trainer
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name, trust_remote_code=script_args.trust_remote_code)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
    tokenizer=tokenizer,
    packing=True
)

trainer.train()

# Save the model
trainer.save_model(script_args.output_dir)
"""
参考

https://huggingface.co/docs/trl/sft_trainer

https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from accelerate import Accelerator
from datasets import load_from_disk 
from functools import wraps
from transformers.modeling_utils import unwrap_model

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, HfArgumentParser, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)

from peft import PeftConfig, PeftModel, get_peft_model
from trl.trainer.utils import neftune_post_forward_hook
from trl import is_xpu_available

"""sample

!python finetune/sft_as_hf.py \
--model_name="if001/tiny_mixtral_ja" \
--tokenizer_name="if001/sentencepiece_ja" \
--train_data_dir="" \
--test_data_dir="" \
--logging_steps=1000 \
--save_total_limit=3 \
--save_steps=1000 \
--batch_size=32 \
--gradient_accumulation_steps=16 \
--trust_remote_code=True \
--output_dir=""
"""

import importlib
def is_peft_available() -> bool:
    return importlib.util.find_spec("peft") is not None

class TrainerWrapped(Trainer):
    """
    wrapped impl    
    https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py

    neftune使うためのwrap
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neftune_noise_alpha = 5

    @wraps(Trainer.train)
    def train(self, *args, **kwargs):
        # Activate neftune right before training.
        if self.neftune_noise_alpha is not None:
            self.model = self._trl_activate_neftune(self.model)

        output = super().train(*args, **kwargs)

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            unwrapped_model = unwrap_model(self.model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                embeddings = unwrapped_model.base_model.model.get_input_embeddings()
            else:
                embeddings = unwrapped_model.get_input_embeddings()

            self.neftune_hook_handle.remove()
            del embeddings.neftune_noise_alpha

        return output
    
    def _trl_activate_neftune(self, model):
        r"""
        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
        Since in transformers Trainer we do have an `_activate_neftune` method, we need to rename this method to avoid conflicts.
        """
        unwrapped_model = unwrap_model(model)
        if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        return model
    

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the tokenizer name"})
    train_data_dir: Optional[str] = field(default="", metadata={"help": ""})
    test_data_dir: Optional[str] = field(default="", metadata={"help": ""})

    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
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
    eval_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of eval steps"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    fp16: Optional[bool] = field(default=False, metadata={"help": "Whether to activate fp16 mixed precision"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "Whether to activate bf16 mixed precision"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    target_modules: Optional[List[str]] = field(default=None, metadata={"help": "Target modules for LoRA adapters"})
    train_data_file: Optional[str] = field(default=1024, metadata={"help": "temp dataset save dir"})
    test_data_file: Optional[str] = field(default=1024, metadata={"help": "temp dataset save dir"})
    from_checkpoint: Optional[str] = field(default="", metadata={"help": "checkpoint dir"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name, trust_remote_code=script_args.trust_remote_code)

# Load the dataset
train_data = load_from_disk(script_args.train_data_dir)
test_data = load_from_disk(script_args.test_data_dir)

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

if script_args.from_checkpoint != "":
    print('load from checkpoint...', script_args.from_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(script_args.from_checkpoint)
else:
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
    per_device_eval_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    eval_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to="none",
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    eval_steps=script_args.eval_steps,
    # push_to_hub=script_args.push_to_hub,
    # hub_model_id=script_args.hub_model_id,
    gradient_checkpointing=script_args.gradient_checkpointing,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    evaluation_strategy="steps",
    save_strategy="steps",
    weight_decay=0.01,
)

# Define the LoraConfig
if script_args.use_peft:
    from peft import LoraConfig
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=script_args.target_modules,
    )
    model = get_peft_model(model, peft_config)
else:
    peft_config = None

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define the Trainer
trainer = TrainerWrapped(
    model=model,
    args=training_args,
    data_collator=data_collator,    
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer
)

resume_from_checkpoint=script_args.from_checkpoint is not None
print("resume_from_checkpoint", resume_from_checkpoint)
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# Save the model
trainer.save_model(script_args.output_dir)
print('done...', script_args.output_dir)